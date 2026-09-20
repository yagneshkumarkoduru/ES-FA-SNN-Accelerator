"""
ES-FA SHD v5 cloud training orchestrator.

Runs the SHD v5 experiment on a single EC2 instance that obtains AWS
credentials from the esfa-shd-runner instance profile (IMDS). No presigned
URLs are used anywhere: presigned URLs signed with aws-login temporary
credentials expire with the session token (observed: under an hour), which
silently broke earlier runs whose uploads happened 3-4 hours after signing.

The instance:
  - downloads the staged code bundle and SHD dataset from S3,
  - uploads /var/log/shd.log and a partial results zip every 10 minutes
    (heartbeat + rescue), so a dead or evicted instance still leaves usable
    artifacts,
  - uploads the final results zip and console log on completion,
  - stops instead of terminating (shutdown behavior = stop), so results stay
    on the EBS volume until the operator runs 'terminate'.

Instance preference: g5.xlarge (NVIDIA A10G, CUDA torch) first, then
c6i.8xlarge / c7i.8xlarge (CPU) as fallbacks. A 9-hour watchdog bounds the run.

IAM setup is idempotent and performed by 'launch': creates the
esfa-shd-runner role, its inline S3 policy (cloud/iam/esfa-shd-runner-policy.json
scoped to the run bucket), and the instance profile if missing.
AmazonSSMManagedInstanceCore is attached so SSM Run Command is available for
live debugging.

Requires boto3 + botocore[crt] (the aws login OAuth credential provider needs
the CRT extra). Use the project-local cloud venv, not another project's:

    .venv-cloud\\Scripts\\python.exe cloud\\run_shd_train.py package
    .venv-cloud\\Scripts\\python.exe cloud\\run_shd_train.py launch
    .venv-cloud\\Scripts\\python.exe cloud\\run_shd_train.py status
    .venv-cloud\\Scripts\\python.exe cloud\\run_shd_train.py collect
    .venv-cloud\\Scripts\\python.exe cloud\\run_shd_train.py terminate

Setup (one-time, if .venv-cloud is missing):
    python -m venv .venv-cloud
    .venv-cloud\\Scripts\\python.exe -m pip install "botocore[crt]" boto3
"""
import argparse
import base64
import json
import pathlib
import shutil
import tarfile
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud"
STATE_PATH = CLOUD / ".state.json"
BUNDLE_DIR = CLOUD / ".bundle"
BUNDLE_PATH = BUNDLE_DIR / "shd_bundle.tgz"

REGION = "us-east-1"
BUCKET = "esfa-shd-runs-969739653654"
FALLBACK_AMI = "ami-025d99823a4caad37"  # Ubuntu 24.04 LTS, us-east-1
SUBNETS = [
    "subnet-0879a02cc37518afb",
    "subnet-0de1d5998132ef926",
    "subnet-0eac15028cb6a014c",
    "subnet-0b0043a151a024c6b",
    "subnet-0de53e7b6d7caa40b",
]
INSTANCE_TYPES = ["g5.xlarge", "c6i.8xlarge", "c7i.8xlarge"]

IAM_ROLE_NAME = "esfa-shd-runner"
IAM_PROFILE_NAME = "esfa-shd-runner"
IAM_POLICY_NAME = "esfa-shd-run-policy"
IAM_DIR = CLOUD / "iam"
IAM_TRUST_PATH = IAM_DIR / "esfa-shd-runner-trust.json"
IAM_POLICY_PATH = IAM_DIR / "esfa-shd-runner-policy.json"
SSM_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"

# Stable S3 keys for the pre-staged SHD dataset (reused across runs).
SHD_S3_TRAIN_KEY = "data/shd_train.h5"
SHD_S3_TEST_KEY = "data/shd_test.h5"

# Local SHD dataset paths (must exist before launch).
SHD_LOCAL_TRAIN = ROOT / "data" / "SHD" / "shd_train.h5"
SHD_LOCAL_TEST = ROOT / "data" / "SHD" / "shd_test.h5"

BUNDLE_FILES = [
    "experiments/benchmark_shd_v12.py",
    "experiments/benchmark_shd_esfa.py",
]

# Driver script written into the bundle as run_shd.sh.
# --data-dir is explicit so the training script always finds the dataset
# regardless of Python working-directory resolution.
DRIVER_SH = """#!/bin/bash
DEVICE="$1"
THREADS="$2"
cd /opt/shd || exit 1
python3 -u experiments/benchmark_shd_v12.py \\
  --hidden 160 --layers 2 --batch-size 64 --epochs 64 --seeds 11,22,33,44 \\
  --lr 1e-2 --lr-min 1e-4 --eps 1e-5 --warmup-epochs 5 --ema-decay 0.999 \\
  --train-drop --tta 4 --tta-drop-time 0.05 --tta-drop-neuron 0.03 \\
  --val-fraction 0.05 --split-seed 20260919 \\
  --checkpoint-every 8 \\
  --device "$DEVICE" --threads "$THREADS" \\
  --data-dir ./data \\
  --output-dir results/shd_v12_valclean
"""

# __RUN_ID__, __BUCKET__, __REGION__ are substituted at launch time.
# All S3 access uses the instance profile via IMDS; no credentials or
# presigned URLs appear in user data.
USER_DATA_TEMPLATE = """#!/bin/bash
set -uo pipefail
exec > /var/log/shd.log 2>&1
echo "ES-FA SHD v5 cloud run __RUN_ID__ starting $(date -u +%FT%TZ)"

# Hard watchdog: shut down after 9 hours even if training hangs.
shutdown -h +540 "watchdog" || true

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip curl gzip zip ca-certificates
pip3 install --break-system-packages --quiet awscli || echo "awscli install failed"
export AWS_DEFAULT_REGION=__REGION__

BUCKET=__BUCKET__
RUN=shd/__RUN_ID__

# Heartbeat + rescue: every 10 minutes upload the log and the latest partial
# results. Credentials come from the instance profile and rotate automatically,
# so uploads keep working for the whole run duration.
(
  while true; do
    sleep 600
    cp /var/log/shd.log /tmp/shd_heartbeat.log 2>/dev/null || true
    aws s3 cp /tmp/shd_heartbeat.log "s3://$BUCKET/$RUN/shd_console.log" --no-progress 2>/dev/null || true
    if [ -d /opt/shd/results ]; then
      rm -f /tmp/shd_results_latest.zip
      (cd /opt/shd && zip -qr /tmp/shd_results_latest.zip results) 2>/dev/null || true
      aws s3 cp /tmp/shd_results_latest.zip "s3://$BUCKET/$RUN/shd_results_latest.zip" --no-progress 2>/dev/null || true
    fi
  done
) &
HEARTBEAT_PID=$!

mkdir -p /opt/shd && cd /opt/shd

if ! aws s3 cp "s3://$BUCKET/$RUN/shd_bundle.tgz" bundle.tgz --no-progress; then
  echo "FATAL: code bundle download failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
if ! tar xzf bundle.tgz; then
  echo "FATAL: code bundle extract failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
echo "Code bundle extracted OK"

echo "Downloading SHD dataset from S3"
mkdir -p data/SHD
if ! aws s3 cp "s3://$BUCKET/data/shd_train.h5" data/SHD/shd_train.h5 --no-progress; then
  echo "FATAL: SHD train set download failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
if ! aws s3 cp "s3://$BUCKET/data/shd_test.h5" data/SHD/shd_test.h5 --no-progress; then
  echo "FATAL: SHD test set download failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
echo "SHD dataset ready: train=$(du -sh data/SHD/shd_train.h5 | cut -f1), test=$(du -sh data/SHD/shd_test.h5 | cut -f1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU present; installing CUDA torch"
  DEVICE=cuda
  THREADS=0
  pip3 install --break-system-packages --quiet torch --index-url https://download.pytorch.org/whl/cu121 || echo "torch install failed"
else
  echo "no GPU; installing CPU torch"
  DEVICE=cpu
  THREADS=$(nproc)
  pip3 install --break-system-packages --quiet torch --index-url https://download.pytorch.org/whl/cpu || echo "torch install failed"
fi
pip3 install --break-system-packages --quiet numpy h5py || echo "deps install failed"
python3 -c "import torch, numpy, h5py; print('imports OK: torch', torch.__version__)" || echo "torch import failed"

bash run_shd.sh "$DEVICE" "$THREADS" || echo "WARNING: training driver exited nonzero"

mkdir -p /opt/out
cp -r results /opt/out/ 2>/dev/null || echo "WARNING: no results directory to copy"
(cd /opt/out && zip -qr /opt/shd_results.zip results) 2>/dev/null || echo "WARNING: final zip failed"

kill "$HEARTBEAT_PID" 2>/dev/null || true
sleep 2
cp /var/log/shd.log /opt/shd_console.log
aws s3 cp /opt/shd_results.zip "s3://$BUCKET/$RUN/shd_results.zip" --no-progress || echo "WARNING: results upload failed"
aws s3 cp /opt/shd_console.log "s3://$BUCKET/$RUN/shd_console.log" --no-progress || echo "WARNING: final log upload failed"

echo "ES-FA SHD v5 done $(date -u +%FT%TZ)"
shutdown -h now
"""


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise SystemExit("No cloud/.state.json; run 'launch' first.")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def cmd_package(_args) -> None:
    stage = BUNDLE_DIR / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rel in BUNDLE_FILES:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"Missing bundle input: {rel}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (stage / "run_shd.sh").write_text(DRIVER_SH, encoding="utf-8", newline="\n")
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with tarfile.open(BUNDLE_PATH, "w:gz") as tar:
        for item in sorted(stage.rglob("*")):
            tar.add(item, arcname=str(item.relative_to(stage)))
    shutil.rmtree(stage)
    print(f"Bundle: {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024:.1f} KB)")


def ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            s3.create_bucket(Bucket=BUCKET)
            print(f"Created bucket: {BUCKET}")
        else:
            raise


def ensure_shd_staged(s3) -> None:
    """Upload SHD h5 files to S3 if not already present. Sourced from local data/SHD/.

    Uses stable S3 keys (data/shd_train.h5 and data/shd_test.h5) shared across
    all runs, so the upload is a one-time cost.
    """
    pairs = [
        (SHD_LOCAL_TRAIN, SHD_S3_TRAIN_KEY, "train"),
        (SHD_LOCAL_TEST,  SHD_S3_TEST_KEY,  "test"),
    ]
    for local_path, s3_key, label in pairs:
        try:
            head = s3.head_object(Bucket=BUCKET, Key=s3_key)
            size_mb = head["ContentLength"] / 1024 / 1024
            print(f"SHD {label} already staged in S3 ({size_mb:.0f} MB): s3://{BUCKET}/{s3_key}")
            continue
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchKey", "403"):
                raise

        if not local_path.exists():
            raise SystemExit(
                f"SHD {label} set not found locally at {local_path} and not in S3.\n"
                "Download from https://compneuro.net/datasets/ (shd_train.h5 and shd_test.h5)\n"
                "and place in data/SHD/ before launching."
            )
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"Staging SHD {label} to S3 ({size_mb:.0f} MB) - this is a one-time upload...")
        s3.upload_file(str(local_path), BUCKET, s3_key)
        print(f"  Staged: s3://{BUCKET}/{s3_key}")


def ensure_runner_profile(iam) -> str:
    """Create the run IAM role, inline policy, and instance profile if missing.

    Idempotent: safe to run on every launch. The inline policy is re-applied
    each time so the deployed permissions always match cloud/iam/.
    """
    try:
        iam.get_role(RoleName=IAM_ROLE_NAME)
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=IAM_TRUST_PATH.read_text(encoding="utf-8"),
            Description="ES-FA SHD cloud runs: read run inputs and write run outputs in the esfa-shd-runs bucket",
            Tags=[
                {"Key": "Project", "Value": "ES-FA"},
                {"Key": "ManagedBy", "Value": "run_shd_train"},
            ],
        )
        print(f"Created IAM role: {IAM_ROLE_NAME}")
    iam.put_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyName=IAM_POLICY_NAME,
        PolicyDocument=IAM_POLICY_PATH.read_text(encoding="utf-8"),
    )
    iam.attach_role_policy(RoleName=IAM_ROLE_NAME, PolicyArn=SSM_POLICY_ARN)
    try:
        profile = iam.get_instance_profile(InstanceProfileName=IAM_PROFILE_NAME)["InstanceProfile"]
        has_role = any(r["RoleName"] == IAM_ROLE_NAME for r in profile.get("Roles", []))
    except iam.exceptions.NoSuchEntityException:
        has_role = False
        iam.create_instance_profile(
            InstanceProfileName=IAM_PROFILE_NAME,
            Tags=[{"Key": "Project", "Value": "ES-FA"}],
        )
        print(f"Created instance profile: {IAM_PROFILE_NAME}")
    if not has_role:
        iam.add_role_to_instance_profile(
            InstanceProfileName=IAM_PROFILE_NAME, RoleName=IAM_ROLE_NAME
        )
        print(f"Added role {IAM_ROLE_NAME} to instance profile {IAM_PROFILE_NAME}")
    return IAM_PROFILE_NAME


def resolve_ami(ssm) -> str:
    try:
        return ssm.get_parameter(
            Name="/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
        )["Parameter"]["Value"]
    except ClientError:
        return FALLBACK_AMI


def cmd_launch(_args) -> None:
    if not BUNDLE_PATH.exists():
        cmd_package(_args)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"shd/{run_id}"
    keys = {
        "bundle_key":  f"{prefix}/shd_bundle.tgz",
        "results_key": f"{prefix}/shd_results.zip",
        "log_key":     f"{prefix}/shd_console.log",
        "latest_key":  f"{prefix}/shd_results_latest.zip",
    }

    s3 = boto3.client("s3", region_name=REGION)
    ensure_bucket(s3)
    ensure_shd_staged(s3)
    s3.upload_file(str(BUNDLE_PATH), BUCKET, keys["bundle_key"])

    iam = boto3.client("iam", region_name=REGION)
    profile_name = ensure_runner_profile(iam)
    print(f"Instance profile ready: {profile_name}")

    user_data = (
        USER_DATA_TEMPLATE
        .replace("__RUN_ID__", run_id)
        .replace("__BUCKET__", BUCKET)
        .replace("__REGION__", REGION)
    )

    ec2 = boto3.client("ec2", region_name=REGION)
    ami = resolve_ami(boto3.client("ssm", region_name=REGION))
    resp = None
    chosen_subnet = None
    chosen_type = None
    last_error = None
    for instance_type in INSTANCE_TYPES:
        for subnet in SUBNETS:
            try:
                resp = ec2.run_instances(
                    ImageId=ami,
                    InstanceType=instance_type,
                    MinCount=1,
                    MaxCount=1,
                    SubnetId=subnet,
                    InstanceInitiatedShutdownBehavior="stop",
                    IamInstanceProfile={"Name": profile_name},
                    BlockDeviceMappings=[
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {
                                "VolumeSize": 60,
                                "VolumeType": "gp3",
                                "DeleteOnTermination": True,
                            },
                        }
                    ],
                    UserData=base64.b64encode(user_data.encode("utf-8")).decode("ascii"),
                    TagSpecifications=[
                        {
                            "ResourceType": "instance",
                            "Tags": [
                                {"Key": "Name",    "Value": f"esfa-shd-v5-{run_id}"},
                                {"Key": "Project", "Value": "ES-FA"},
                                {"Key": "Purpose", "Value": "shd-v5-innovation"},
                            ],
                        }
                    ],
                )
                chosen_subnet = subnet
                chosen_type = instance_type
                break
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in (
                    "InsufficientInstanceCapacity",
                    "Unsupported",
                    "VcpuLimitExceeded",
                    "MaxSpotInstanceCountExceeded",
                ):
                    print(f"{code} for {instance_type} in {subnet}, trying next")
                    last_error = exc
                    continue
                raise
        if resp is not None:
            break
    if resp is None:
        raise SystemExit(f"No capacity / quota for any instance type: {last_error}")

    instance_id = resp["Instances"][0]["InstanceId"]
    STATE_PATH.write_text(
        json.dumps(
            {
                "run_id":        run_id,
                "instance_id":   instance_id,
                "instance_type": chosen_type,
                "subnet":        chosen_subnet,
                "bucket":        BUCKET,
                **keys,
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Launched {chosen_type} instance: {instance_id} (subnet {chosen_subnet})")
    print(f"  run id  : {run_id}")
    print(f"  AMI     : {ami}")
    print(f"  profile : {profile_name} (S3 access via IMDS; SSM Run Command enabled)")
    print(f"  SHD     : s3://{BUCKET}/{SHD_S3_TRAIN_KEY} + {SHD_S3_TEST_KEY}")
    print("Watchdog: 9 hours. Heartbeat (log + results) every 10 min.")
    print("Check:   .venv-cloud\\Scripts\\python.exe cloud\\run_shd_train.py status")
    print(f"SSM:     aws ssm describe-instance-information --filters Key=InstanceIds,Values={instance_id}")


def cmd_status(_args) -> None:
    state = load_state()
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        inst = ec2.describe_instances(
            InstanceIds=[state["instance_id"]]
        )["Reservations"][0]["Instances"][0]
        print(f"Instance {state['instance_id']} ({state['instance_type']}): {inst['State']['Name']}")
    except (ClientError, IndexError, KeyError):
        print(f"Instance {state['instance_id']}: gone (terminated and cleaned up)")
    s3 = boto3.client("s3", region_name=REGION)
    resp = s3.list_objects_v2(Bucket=state["bucket"], Prefix=f"shd/{state['run_id']}/")
    objects = resp.get("Contents", [])
    if objects:
        for obj in objects:
            size_kb = obj["Size"] / 1024
            print(f"  {obj['Key']:<60} {size_kb:>10.1f} KB   {obj['LastModified']}")
    else:
        print("  (no output files in S3 yet)")


def cmd_collect(_args) -> None:
    state = load_state()
    s3 = boto3.client("s3", region_name=REGION)
    source_key = None
    try:
        s3.head_object(Bucket=state["bucket"], Key=state["results_key"])
        source_key = state["results_key"]
        print("Canonical results zip found.")
    except ClientError:
        latest_key = state.get("latest_key", f"shd/{state['run_id']}/shd_results_latest.zip")
        try:
            s3.head_object(Bucket=state["bucket"], Key=latest_key)
            source_key = latest_key
            print("Canonical results not present; collecting latest heartbeat snapshot.")
        except ClientError:
            print("Results not uploaded yet; check status.")
            return
    out_dir = ROOT / "results" / f"cloud_shd_v5_{state['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = BUNDLE_DIR / f"shd_results_{state['run_id']}.zip"
    print(f"Downloading s3://{state['bucket']}/{source_key} ...")
    s3.download_file(state["bucket"], source_key, str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted to: {out_dir}")
    try:
        log_dest = out_dir / "shd_console.log"
        s3.download_file(state["bucket"], state["log_key"], str(log_dest))
        print(f"Console log: {log_dest}")
    except ClientError:
        print("Console log not yet uploaded.")
    for json_file in sorted(out_dir.rglob("*.json")):
        print(f"  {json_file.relative_to(out_dir)}")


def cmd_terminate(_args) -> None:
    state = load_state()
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        name = ec2.describe_instances(
            InstanceIds=[state["instance_id"]]
        )["Reservations"][0]["Instances"][0]["State"]["Name"]
    except (ClientError, IndexError, KeyError):
        print("Instance already gone.")
        return
    if name in ("terminated", "shutting-down"):
        print(f"Instance already {name}.")
        return
    ec2.terminate_instances(InstanceIds=[state["instance_id"]])
    print(f"Termination requested for {state['instance_id']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ES-FA SHD v5 cloud training")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("package")
    sub.add_parser("launch")
    sub.add_parser("status")
    sub.add_parser("collect")
    sub.add_parser("terminate")
    args = parser.parse_args()
    handlers = {
        "package":   cmd_package,
        "launch":    cmd_launch,
        "status":    cmd_status,
        "collect":   cmd_collect,
        "terminate": cmd_terminate,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
