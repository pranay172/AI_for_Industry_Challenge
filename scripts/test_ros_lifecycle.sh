#!/usr/bin/env bash
# Isolated local ROS graph; no evaluator or robot controller is launched.
set -eu
ulimit -c 0
cd "$(dirname "$0")/.."
export AMENT_PREFIX_PATH="$PWD/.pixi/envs/default"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_DOMAIN_ID=86
export ZENOH_ROUTER_CHECK_ATTEMPTS=10
export ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/127.0.0.1:17447"];listen/endpoints=[];scouting/multicast/enabled=false;transport/shared_memory/enabled=false'
test_logs=$(mktemp -d /tmp/aic-lifecycle.XXXXXX)
ZENOH_CONFIG_OVERRIDE='listen/endpoints=["tcp/127.0.0.1:17447"];connect/endpoints=[];scouting/multicast/enabled=false;transport/shared_memory/enabled=false' \
  .pixi/envs/default/lib/rmw_zenoh_cpp/rmw_zenohd > "$test_logs/router.log" 2>&1 &
router_pid=$!
trap 'kill "$router_pid" 2>/dev/null || true; wait "$router_pid" 2>/dev/null || true' EXIT
printf 'Router log: %s/router.log\n' "$test_logs"
AIC_ROS_INTEGRATION=1 PYTHONDONTWRITEBYTECODE=1 \
  .pixi/envs/default/bin/python -m pytest tests/test_model_lifecycle.py -q
