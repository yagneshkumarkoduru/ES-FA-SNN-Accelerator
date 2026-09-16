"""
ES-FA SHD v5 cloud training orchestrator.

Runs the SHD v5 experiment (ES-FA innovations on the protocol-corrected
backbone) on a single EC2 instance with no IAM instance profile. The
instance receives time-limited presigned S3 URLs via user data for the code
bundle (GET) and the results archive and console log (PUTs); the SHD dataset
is downloaded on the instance from the public source (compneuro.net). The
instance self-terminates when training finishes.

Instance preference: g5.xlarge (NVIDIA A10G, CUDA torch) first, then
c6i.8xlarge / c7i.8xlarge (CPU) as fallbacks if GPU capacity or quota is
unavailable. A 6-hour watchdog bounds the run.

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
PRESIGN_EXPIRY = 604800

BUNDLE_FILES = [
    "experiments/benchmark_shd_v5.py",
    "experiments/benchmark_shd_esfa.py",
]

DRIVER_SH = """#!/bin/bash
DEVICE="$1"
THREADS="$2"
cd /opt/shd || exit 1
python3 -u experiments/benchmark_shd_v5.py \\
  --hidden 512 --batch-size 256 --epochs 160 --seeds 42,123 \\
  --governor --dual-readout --device "$DEVICE" --threads "$THREADS" \\
  --checkpoint-every 2 --log-every 5 \\
  --output-dir results/shd_v5_innovation
"""

USER_DATA_TEMPLATE = """#!/bin/bash
set -uo pipefail
exec > /var/log/shd.log 2>&1
echo "ES-FA SHD v5 cloud run __RUN_ID__ starting $(date -u +%FT%TZ)"
shutdown -h +360 "watchdog" || true
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip curl gzip zip ca-certificates
mkdir -p /opt/shd && cd /opt/shd
curl -fsSL -o bundle.tgz "__GET_URL__" && tar xzf bundle.tgz || echo "bundle download/extract failed"
echo "downloading SHD dataset"
curl -fsSL -o shd_train.h5.gz https://compneuro.net/datasets/shd_train.h5.gz && gzip -d shd_train.h5.gz || echo "train download failed"
curl -fsSL -o shd_test.h5.gz https://compneuro.net/datasets/shd_test.h5.gz && gzip -d shd_test.h5.gz || echo "test download failed"
mkdir -p data/SHD
mv shd_train.h5 shd_test.h5 data/SHD/ 2>/dev/null || echo "dataset move failed"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU present; installing CUDA torch"
  pip3 install --break-system-packages --quiet torch --index-url https://download.pytorch.org/whl/cu121 || echo "torch install failed"
  DEVICE=cuda
  THREADS=0
else
  echo "no GPU; installing CPU torch"
  pip3 install --break-system-packages --quiet torch --index-url https://download.pytorch.org/whl/cpu || echo "torch install failed"
  DEVICE=cpu
  THREADS=$(nproc)
fi
pip3 install --break-system-packages --quiet numpy h5py || echo "deps install failed"
bash run_shd.sh "$DEVICE" "$THREADS" || echo "driver exited nonzero"
mkdir -p /opt/out
cp -r results /opt/out/ 2>/dev/null || echo "no results dir"
cd /opt/out && zip -qr /opt/shd_results.zip results 2>/dev/null || echo "zip failed"
cp /var/log/shd.log /opt/shd_console.log
curl -fsSL -T /opt/shd_results.zip "__PUT_RESULTS__" || echo "results upload failed"
curl -fsSL -T /opt/shd_console.log "__PUT_LOG__" || echo "log upload failed"
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
        "bundle_key": f"{prefix}/shd_bundle.tgz",
        "results_key": f"{prefix}/shd_results.zip",
        "log_key": f"{prefix}/shd_console.log",
    }

    s3 = boto3.client("s3", region_name=REGION)
    ensure_bucket(s3)
    s3.upload_file(str(BUNDLE_PATH), BUCKET, keys["bundle_key"])

    get_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": keys["bundle_key"]},
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
        USER_DATA_TEMPLATE.replace("__RUN_ID__", run_id)
        .replace("__GET_URL__", get_url)
        .replace("__PUT_RESULTS__", put_results)
        .replace("__PUT_LOG__", put_log)
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
                    InstanceMarketOptions={
                        "MarketType": "spot",
                        "SpotOptions": {
                            "SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate",
                        },
                    },
                    UserData=base64.b64encode(user_data.encode("utf-8")).decode("ascii"),
                    TagSpecifications=[
                        {
                            "ResourceType": "instance",
                            "Tags": [
                                {"Key": "Name", "Value": f"esfa-shd-v5-{run_id}"},
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
        raise SystemExit(f"No spot capacity / quota for any instance type: {last_error}")

    instance_id = resp["Instances"][0]["InstanceId"]
    STATE_PATH.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "instance_type": chosen_type,
                "subnet": chosen_subnet,
                "bucket": BUCKET,
                **keys,
                "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Launched {chosen_type} spot instance: {instance_id} (subnet {chosen_subnet})")
    print(f"  run id : {run_id}")
    print(f"  AMI    : {ami}")
    print("Watchdog: 6 hours. Check progress: py -3 cloud/run_shd_train.py status")


def cmd_status(_args) -> None:
    state = load_state()
    ec2 = boto3.client("ec2", region_name=REGION)
    inst = ec2.describe_instances(InstanceIds=[state["instance_id"]])["Reservations"][0]["Instances"][0]
    print(f"Instance {state['instance_id']} ({state['instance_type']}): {inst['State']['Name']}")
    s3 = boto3.client("s3", region_name=REGION)
    resp = s3.list_objects_v2(Bucket=state["bucket"], Prefix=f"shd/{state['run_id']}/")
    for obj in resp.get("Contents", []):
        print(f"  {obj['Key']} ({obj['Size']} bytes, {obj['LastModified']})")


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
    s3.download_file(state["bucket"], state["results_key"], str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted to: {out_dir}")
    try:
        s3.download_file(state["bucket"], state["log_key"], str(out_dir / "shd_console.log"))
        print(f"Console log: {out_dir / 'shd_console.log'}")
    except ClientError:
        print("Console log not uploaded (yet).")
    for json_file in sorted(out_dir.rglob("*.json")):
        print(f"  {json_file.relative_to(out_dir)}")


def cmd_terminate(_args) -> None:
    state = load_state()
    ec2 = boto3.client("ec2", region_name=REGION)
    name = ec2.describe_instances(InstanceIds=[state["instance_id"]])["Reservations"][0]["Instances"][0]["State"]["Name"]
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
        "package": cmd_package,
        "launch": cmd_launch,
        "status": cmd_status,
        "collect": cmd_collect,
        "terminate": cmd_terminate,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
