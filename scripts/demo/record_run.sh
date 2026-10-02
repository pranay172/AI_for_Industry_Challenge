#!/bin/bash
# Run one benchmark run (scripts/benchmark.py) and record it through a view-only
# Gazebo GUI attached to the evaluator container.
#
# The evaluator's own launch is unchanged (gazebo_gui:=false). Once it is up, a
# separate `gz sim -g` client with demo_gui.config renders to an Xvfb display
# inside the evaluator container, and capture_frames.py saves that display at
# 8 fps to <run>/demo_frames. The viewer only subscribes to the scene; it does
# lower the simulation's real-time factor somewhat.
#
# Usage: scripts/demo/record_run.sh <run directory prepared with benchmark.py prepare>
set -u
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
DEMO="$ROOT/scripts/demo"
RUN=$(readlink -f "$1")
PY=${PY:-$ROOT/.pixi/envs/default/bin/python}
EVAL_IMAGE=${EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}
# Elevated three-quarter view of the board area, looking towards the robot.
POSE='pose: {position: {x: 0.532, y: -0.475, z: 1.578}, orientation: {w: 0.389, x: -0.2784, y: 0.1246, z: 0.8693}}'
cd "$ROOT"
"$PY" scripts/benchmark.py run "$RUN" --eval-image "$EVAL_IMAGE" --gpu > "$RUN/runner.log" 2>&1 &
RPID=$!
C=""
until [ -n "$C" ]; do
  kill -0 $RPID 2>/dev/null || { echo "runner exited before the evaluator started"; wait $RPID; exit 1; }
  sleep 2
  C=$(docker ps --filter name=aic-baseline- --format '{{.Names}}' | grep -- '-eval-' | head -1)
done
echo "evaluator container $C"
docker cp "$DEMO/demo_gui.config" "$C:/tmp/demo_gui.config"
docker cp "$DEMO/capture_frames.py" "$C:/tmp/capture_frames.py"
docker exec "$C" mkdir -p /tmp/fb
docker exec -d "$C" bash -c "Xvfb :99 -nocursor -screen 0 960x540x24 -fbdir /tmp/fb -nolisten tcp > /tmp/xvfb.log 2>&1"
until docker exec "$C" bash -c ". /ws_aic/install/setup.bash; gz service -l 2>/dev/null | grep -q /world/aic_world/scene/info"; do
  kill -0 $RPID 2>/dev/null || break; sleep 2; done
# No software-GL override: the GUI renders on the GPU through EGL, like the camera sensors.
docker exec -d "$C" bash -c ". /ws_aic/install/setup.bash; DISPLAY=:99 gz sim -g --gui-config /tmp/demo_gui.config > /tmp/gui.log 2>&1"
until docker exec "$C" bash -c ". /ws_aic/install/setup.bash; gz service -l 2>/dev/null | grep -q /gui/move_to/pose"; do
  kill -0 $RPID 2>/dev/null || break; sleep 2; done
sleep 3
docker exec "$C" bash -c ". /ws_aic/install/setup.bash; gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera --reptype gz.msgs.Boolean --timeout 3000 --req '$POSE'"
# Frames are written as the invoking user, into the run's results mount.
docker exec -d -u "$(id -u):$(id -g)" "$C" python3 /tmp/capture_frames.py --out /results/demo_frames --fps 8
echo "recording started $(date -Is)"
wait $RPID; status=$?
mv "$RUN/results/demo_frames" "$RUN/demo_frames" 2>/dev/null
echo "runner exit $status; frames $(ls "$RUN/demo_frames" 2>/dev/null | wc -l)"
exit $status
