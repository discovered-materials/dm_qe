#!/usr/bin/env bash
# runner-setup.sh — make a GPU machine a self-hosted GitHub Actions runner for the GPU tests workflow.
# Run as root on a machine that has the toolchain (gpu-quickstart/cloud/setup_node.sh) and an NVIDIA
# driver; idempotent.
#
#   sudo .github/ci/runner-setup.sh <registration-token> [repository-url]
#
# The token comes from the repository (Settings -> Actions -> Runners -> New self-hosted runner,
# the value after --token; valid for an hour). What it sets up:
#   github-runner      the unprivileged user the jobs run as
#   /opt/actions-runner  the runner, as a systemd service, labels self-hosted,gpu,dm-qe (RUNNER_LABELS)
#   /opt/qe-ci/build   persistent build directories (incremental builds between runs)
#   cmake              the 3.31.6 of setup_node.sh on everyone's PATH
#   idle watchdog      powers the machine off after IDLE_MINUTES (20) without a job and never in the
#                      first 20 minutes after boot; a stopped cloud VM does not bill, the workflow
#                      starts it again. `touch /opt/qe-ci/keep-alive` keeps the machine up.
set -euo pipefail
TOKEN=${1:?usage: runner-setup.sh <registration-token> [repository-url]}
URL=${2:-https://github.com/discovered-materials/dm_qe}
LABELS=${RUNNER_LABELS:-self-hosted,gpu,dm-qe}; IDLE=${IDLE_MINUTES:-20}
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
id github-runner > /dev/null 2>&1 || useradd -m -s /bin/bash github-runner
install -d -o github-runner -g github-runner /opt/qe-ci /opt/qe-ci/build
for c in "$HERE"/../../gpu-quickstart/tools/cmake-*-linux-x86_64/bin/cmake /opt/qe-ci/gpu-quickstart/tools/cmake-*-linux-x86_64/bin/cmake; do
  [ -x "$c" ] && ln -sfn "$(realpath "$c")" /usr/local/bin/cmake && break
done
RUN=/opt/actions-runner
if [ ! -x "$RUN/config.sh" ]; then
  V=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')
  echo "== actions runner $V"
  install -d -o github-runner -g github-runner "$RUN"
  curl -fsSL "https://github.com/actions/runner/releases/download/v$V/actions-runner-linux-x64-$V.tar.gz" | sudo -u github-runner tar -xzf - -C "$RUN"
  "$RUN/bin/installdependencies.sh" > /dev/null
fi
if [ ! -f "$RUN/.runner" ]; then
  echo "== registering $(hostname -s) at $URL with labels $LABELS"
  sudo -u github-runner "$RUN/config.sh" --unattended --url "$URL" --token "$TOKEN" --name "$(hostname -s)" --labels "$LABELS" --work _work --replace
fi
( cd "$RUN" && ( ./svc.sh status > /dev/null 2>&1 || ./svc.sh install github-runner > /dev/null ) && ./svc.sh start > /dev/null 2>&1 || true )
cat > /opt/qe-ci/idle-shutdown.sh <<W
#!/bin/bash
# powers the machine off after $IDLE minutes without a job (no Runner.Worker process), never in the
# first $IDLE minutes after boot, never while /opt/qe-ci/keep-alive exists
S=/run/qe-ci-last-busy
[ -f /opt/qe-ci/keep-alive ] && exit 0
if pgrep -f 'Runner.Worker' > /dev/null; then touch \$S; exit 0; fi
up=\$(( \$(cut -d. -f1 /proc/uptime) / 60 )); [ \$up -ge $IDLE ] || exit 0
[ -f \$S ] && [ \$(( (\$(date +%s) - \$(stat -c %Y \$S)) / 60 )) -lt $IDLE ] && exit 0
logger -t qe-ci "idle for $IDLE minutes: powering off"
systemctl poweroff
W
chmod 755 /opt/qe-ci/idle-shutdown.sh
echo '*/5 * * * * root /opt/qe-ci/idle-shutdown.sh' > /etc/cron.d/qe-ci-idle-shutdown
echo "== runner: $(cd "$RUN" && ./svc.sh status 2>/dev/null | grep -E 'Active:' | head -1)"
echo "RUNNER OK: $(hostname -s), labels $LABELS, idle shutdown after $IDLE min"
