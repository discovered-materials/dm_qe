#!/usr/bin/env bash
# create_vm.sh — a Google Compute Engine VM with the shape the published timings were taken on.
# Uses your own gcloud login and project (gcloud auth login; gcloud config set project …).
#
#   VM=qe-h100 ZONE=us-central1-a MACHINE_TYPE=a3-highgpu-4g benchmarks/cloud/create_vm.sh
#
# a3-highgpu-4g (4x H100 80 GB, 104 vCPUs, 936 GB) is the measured machine; a3-highgpu-1g (one H100,
# 26 vCPUs) gives the same per-GPU timings at a quarter of the price, with one case at a time.
# The image is the Deep Learning VM image the node runs (Ubuntu 22.04, NVIDIA driver 580, CUDA 13
# capable). H100 capacity usually comes through Dynamic Workload Scheduler flex-start: the VM waits in
# a queue (minutes to hours), runs at most MAX_RUN, then stops; its local SSDs are discarded at the
# stop and its boot disk stays. ON_DEMAND=1 asks for a regular VM instead. The VM bills while it runs:
# gcloud compute instances stop "$VM" --zone "$ZONE" when done.
set -euo pipefail
VM=${VM:-qe-h100}; ZONE=${ZONE:-us-central1-a}
MACHINE_TYPE=${MACHINE_TYPE:-a3-highgpu-4g}
IMAGE=${IMAGE:-common-cu129-ubuntu-2204-nvidia-580-v20261001}; IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
MAX_RUN=${MAX_RUN:-393000s}        # flex-start limit, about 4.5 days
BOOT_GB=${BOOT_GB:-300}
case "$MACHINE_TYPE" in a3-highgpu-1g) NGPU=1;; a3-highgpu-2g) NGPU=2;; a3-highgpu-4g) NGPU=4;; a3-highgpu-8g) NGPU=8;;
  *) echo "MACHINE_TYPE $MACHINE_TYPE: this script knows the a3-highgpu machines (H100 80 GB)" >&2; exit 2;; esac
if gcloud compute instances describe "$VM" --zone="$ZONE" > /dev/null 2>&1; then
  echo "$VM exists in $ZONE ($(gcloud compute instances describe "$VM" --zone="$ZONE" --format='value(status)'))"; exit 0
fi
ARGS=(--zone="$ZONE" --machine-type="$MACHINE_TYPE" --accelerator=type=nvidia-h100-80gb,count=$NGPU
      --maintenance-policy=TERMINATE --image="$IMAGE" --image-project="$IMAGE_PROJECT"
      --boot-disk-size="${BOOT_GB}GB" --boot-disk-type=pd-balanced --metadata=enable-oslogin=TRUE)
for _ in $(seq 1 $((2*NGPU))); do ARGS+=(--local-ssd=interface=NVME); done   # the a3 machines come with two per GPU
if [ "${ON_DEMAND:-0}" != 1 ]; then
  ARGS+=(--provisioning-model=FLEX_START --max-run-duration="$MAX_RUN" --instance-termination-action=STOP
         --discard-local-ssds-at-termination-timestamp=true)
fi
echo "creating $VM: $MACHINE_TYPE, $NGPU x H100 80 GB, $IMAGE$([ "${ON_DEMAND:-0}" = 1 ] || echo ", flex-start (queued for capacity)")"
gcloud compute instances create "$VM" "${ARGS[@]}"
cat <<MSG

Next, on the VM (gcloud compute ssh $VM --zone $ZONE):
  git clone <this repository> && cd <it>
  sudo benchmarks/cloud/setup_node.sh           # toolchain (about 20 minutes, NVHPC is large)
  benchmarks/scripts/fetch_pseudos.sh
and then benchmarks/README.md, "Running".
MSG
