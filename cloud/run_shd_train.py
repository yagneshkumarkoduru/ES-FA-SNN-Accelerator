"""
ES-FA SHD v5 cloud training orchestrator.

Runs the SHD v5 experiment on a single EC2 instance. The instance receives
time-limited presigned S3 URLs via user data for the code bundle (GET), the
pre-staged SHD dataset (GET), and the results archive + console log (PUTs).
The instance self-terminates when training finishes.

Instance preference: g5.xlarge (NVIDIA A10G, CUDA torch) first, then
c6i.8xlarge / c7i.8xlarge (CPU) as fallbacks. A 6-hour watchdog bounds the run.

SHD dataset is pre-staged to S3 during the launch step (sourced from the local
data/SHD/ directory). This avoids depending on the external compneuro.net server
at training time, which was the root cause of previous silent failures.

A background heartbeat uploads /var/log/shd.log to S3 every 30 minutes so the
run is always diagnosable even if the final upload fails.

Usage:
    py -3 cloud/run_shd_train.py package
    py -3 cloud/run_shd_train.py launch
    py -3 cloud/run_shd_train.py status
    py -3 cloud/run_shd_train.py collect
    py -3 cloud/run_shd_train.py terminate
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
PRESIGN_EXPIRY = 604800  # 7 days

# Stable S3 keys for the pre-staged SHD dataset (reused across runs).
SHD_S3_TRAIN_KEY = "data/shd_train.h5"
SHD_S3_TEST_KEY = "data/shd_test.h5"

# Local SHD dataset paths (must exist before launch).
SHD_LOCAL_TRAIN = ROOT / "data" / "SHD" / "shd_train.h5"
SHD_LOCAL_TEST = ROOT / "data" / "SHD" / "shd_test.h5"

BUNDLE_FILES = [
    "experiments/benchmark_shd_v5.py",
    "experiments/benchmark_shd_esfa.py",
]

# Driver script written into the bundle as run_shd.sh.
# --data-dir is explicit so the training script always finds the dataset
# regardless of Python working-directory resolution.
DRIVER_SH = """#!/bin/bash
DEVICE="$1"
THREADS="$2"
cd /opt/shd || exit 1
python3 -u experiments/benchmark_shd_v5.py \\
  --hidden 512 --batch-size 256 --epochs 160 --seeds 42,123 \\
  --governor --dual-readout --device "$DEVICE" --threads "$THREADS" \\
  --checkpoint-every 2 --log-every 5 \\
  --data-dir ./data \\
  --output-dir results/shd_v5_innovation
"""

# __MARKERS__ replaced at launch time; double-quoted URLs are safe in bash
# because & inside double quotes is treated as a literal character.
USER_DATA_TEMPLATE = """#!/bin/bash
set -uo pipefail
exec > /var/log/shd.log 2>&1
echo "ES-FA SHD v5 cloud run __RUN_ID__ starting $(date -u +%FT%TZ)"

# Hard watchdog: shut down after 6 hours even if training hangs.
shutdown -h +360 "watchdog" || true

# Background heartbeat: snapshot and upload the log to S3 every 30 minutes.
# This guarantees log visibility even if the final upload fails.
(
  while true; do
    sleep 1800
    cp /var/log/shd.log /tmp/shd_heartbeat.log 2>/dev/null || true
    curl -fsSL -T /tmp/shd_heartbeat.log "__PUT_LOG__" 2>/dev/null || true
  done
) &
HEARTBEAT_PID=$!

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip curl gzip zip ca-certificates

mkdir -p /opt/shd && cd /opt/shd

# -- Code bundle (from S3, same-region, reliable) ----------------------------
if ! curl -fsSL -o bundle.tgz "__GET_URL__"; then
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

# -- SHD dataset (from S3, pre-staged, reliable) ------------------------------
echo "Downloading SHD dataset from S3"
mkdir -p data/SHD
if ! curl -fsSL -o data/SHD/shd_train.h5 "__GET_TRAIN__"; then
  echo "FATAL: SHD train set download failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
if ! curl -fsSL -o data/SHD/shd_test.h5 "__GET_TEST__"; then
  echo "FATAL: SHD test set download failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
TRAIN_SIZE=$(du -sh data/SHD/shd_train.h5 | cut -f1)
TEST_SIZE=$(du -sh data/SHD/shd_test.h5 | cut -f1)
echo "SHD dataset ready: train=${TRAIN_SIZE}, test=${TEST_SIZE}"

# -- PyTorch + dependencies ---------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU present; installing CUDA torch"
  DEVICE=cuda
  THREADS=0
  pip3 install --break-system-packages --quiet torch --index-url https://download.pytorch.org/whl/cu121
else
  echo "No GPU; installing CPU torch"
  DEVICE=cpu
  THREADS=$(nproc)
  pip3 install --break-system-packages --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
if [ $? -ne 0 ]; then
  echo "FATAL: torch install failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
if ! pip3 install --break-system-packages --quiet numpy h5py; then
  echo "FATAL: numpy/h5py install failed"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit 1
fi
python3 -c "import torch, numpy, h5py; print('imports OK: torch', torch.__version__)"

# -- Training (non-fatal: upload whatever results exist if training fails) ---
bash run_shd.sh "$DEVICE" "$THREADS" || echo "WARNING: training driver exited nonzero"

# -- Package and upload results -----------------------------------------------
mkdir -p /opt/out
cp -r results /opt/out/ 2>/dev/null || echo "WARNING: no results directory to copy"
cd /opt/out && zip -qr /opt/shd_results.zip results 2>/dev/null || echo "WARNING: zip failed"
cp /var/log/shd.log /opt/shd_console.log

curl -fsSL -T /opt/shd_results.zip "__PUT_RESULTS__" || echo "WARNING: results upload failed"

# Kill heartbeat and do the final complete log upload.
kill "$HEARTBEAT_PID" 2>/dev/null || true
sleep 2
cp /var/log/shd.log /opt/shd_console.log
curl -fsSL -T /opt/shd_console.log "__PUT_LOG__" || echo "WARNING: final log upload failed"

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
        # Check if already in S3.
        try:
            head = s3.head_object(Bucket=BUCKET, Key=s3_key)
            size_mb = head["ContentLength"] / 1024 / 1024
            print(f"SHD {label} already staged in S3 ({size_mb:.0f} MB): s3://{BUCKET}/{s3_key}")
            continue
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchKey", "403"):
                raise

        # Upload from local data directory.
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
    }

    s3 = boto3.client("s3", region_name=REGION)
    ensure_bucket(s3)

    # Stage SHD dataset to S3 (no-op if already there).
    ensure_shd_staged(s3)

    # Upload code bundle.
    s3.upload_file(str(BUNDLE_PATH), BUCKET, keys["bundle_key"])

    # Generate presigned URLs.
    get_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": keys["bundle_key"]},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    get_train = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": SHD_S3_TRAIN_KEY},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    get_test = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": SHD_S3_TEST_KEY},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    put_results = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": keys["results_key"]},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    put_log = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": keys["log_key"]},
        ExpiresIn=PRESIGN_EXPIRY,
    )

    user_data = (
        USER_DATA_TEMPLATE
        .replace("__RUN_ID__", run_id)
        .replace("__GET_URL__",    get_url)
        .replace("__GET_TRAIN__",  get_train)
        .replace("__GET_TEST__",   get_test)
        .replace("__PUT_RESULTS__", put_results)
        .replace("__PUT_LOG__",    put_log)
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
                    InstanceInitiatedShutdownBehavior="terminate",
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
    print(f"Launched {chosen_type} on-demand instance: {instance_id} (subnet {chosen_subnet})")
    print(f"  run id : {run_id}")
    print(f"  AMI    : {ami}")
    print(f"  SHD    : s3://{BUCKET}/{SHD_S3_TRAIN_KEY} + {SHD_S3_TEST_KEY}")
    print("Watchdog: 6 hours. Heartbeat log: every 30 min. Check: py -3 cloud/run_shd_train.py status")


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
            print(f"  {obj['Key']:<60} {size_kb:>8.1f} KB   {obj['LastModified']}")
    else:
        print("  (no output files in S3 yet)")


def cmd_collect(_args) -> None:
    state = load_state()
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.head_object(Bucket=state["bucket"], Key=state["results_key"])
    except ClientError:
        print("Results not uploaded yet; check status.")
        return
    out_dir = ROOT / "results" / f"cloud_shd_v5_{state['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = BUNDLE_DIR / f"shd_results_{state['run_id']}.zip"
    print(f"Downloading results to {out_dir} ...")
    s3.download_file(state["bucket"], state["results_key"], str(zip_path))
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
