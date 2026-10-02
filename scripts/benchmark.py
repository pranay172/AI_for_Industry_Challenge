#!/usr/bin/env python3
"""Freeze and run an isolated AIC benchmark; scoring never enters the policy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "sfp": "aic_model/models/sfp_port_detector.pt",
    "sc": "aic_model/models/sc_port_detector.pt",
    "plug_sfp": "aic_model/models/sfp_plug_pose.pt",
    "plug_sc": "aic_model/models/sc_plug_pose.pt",
}
# Exactly the inputs copied by the model Dockerfile, including untracked weights.
BUILD_INPUTS = [
    "aic_example_policies", "aic_model", "aic_interfaces", "aic_utils",
    "pixi.toml", "pixi.lock", "pixi_env_setup.sh", "docker/aic_model",
    "scripts", "benchmarks", "tests", "pytest.ini",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def output(args, **kwargs):
    return command(args, stdout=subprocess.PIPE, **kwargs).stdout.strip()


CHECKPOINT_MOUNT = "/checkpoints"
CHECKPOINT_VARIABLES = {"sc": "AIC_SC_PORT_DETECTOR_PATH", "sfp": "AIC_SFP_DETECTOR_PATH",
                        "plug_sfp": "AIC_PLUG_POSE_SFP_PATH", "plug_sc": "AIC_PLUG_POSE_SC_PATH"}


def prepare(config, destination, checkpoints=None, capture=True):
    """Freeze source, configuration and any candidate checkpoints (kind -> path)."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Run directory already exists: {destination}")
    config = Path(config).resolve()
    scene = yaml.safe_load(config.read_text())
    if not isinstance(scene, dict) or not scene.get("trials"):
        raise ValueError("Configuration must contain nonempty trials")
    for path in MODEL_PATHS.values():
        if not (ROOT / path).is_file():
            raise FileNotFoundError(f"Required checkpoint missing: {path}")
    expected = json.loads((ROOT / "benchmarks/models.json").read_text())
    for kind, path in MODEL_PATHS.items():
        if sha256(ROOT / path) != expected[kind]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}; version the new artifact first")
    destination.mkdir(parents=True)
    source = destination / "source"
    source.mkdir()
    for item in BUILD_INPUTS:
        src, dst = ROOT / item, source / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", ".pixi"))
        else:
            shutil.copy2(src, dst)
    shutil.copy2(config, destination / "config.yaml")
    for subdir in ("results", "captures"):
        (destination / subdir).mkdir()
        # The official evaluator runs as root inside its container.
        (destination / subdir).chmod(0o777)
    files = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "code_revision": output(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "git_status": output(["git", "status", "--porcelain"], cwd=ROOT),
        "source_sha256": fingerprint,
        "source_files": files,
        "config_sha256": sha256(destination / "config.yaml"),
        "config_origin": str(config),
        "models": expected,
        "acl_enabled": True,
        "ground_truth": False,
        "capture_enabled": bool(capture),
        "expected_trials": list(scene["trials"]),
    }
    if checkpoints:
        # Candidates are copied into the run and mounted read-only in place of the
        # image defaults; hashes make the evaluated weights auditable.
        (destination / "checkpoints").mkdir()
        manifest["checkpoints"] = {}
        for kind, path in checkpoints.items():
            path = Path(path).resolve()
            name = f"{kind}-{path.name}"
            shutil.copy2(path, destination / "checkpoints" / name)
            manifest["checkpoints"][kind] = {"origin": str(path), "name": name,
                                             "sha256": sha256(destination / "checkpoints" / name)}
    origin = ROOT / "benchmarks/qualification.source.json"
    if manifest["config_sha256"] == json.loads(origin.read_text())["sha256"]:
        manifest["qualification_source"] = json.loads(origin.read_text())
    write_json(destination / "manifest.json", manifest)
    (destination / "working-tree.patch").write_text(output(["git", "diff", "HEAD", "--binary"], cwd=ROOT))
    return destination


def summarize(scoring, expected_trials):
    """Parse the engine's actual TierScore serialization, not policy log claims."""
    if not isinstance(scoring, dict) or "total" not in scoring:
        raise ValueError("Missing official scoring total")
    total = float(scoring["total"])
    if not math.isfinite(total):
        raise ValueError("Non-finite score")
    trials, breakdown = [], Counter()
    for name in expected_trials:
        if name not in scoring:
            trials.append({"trial": name, "outcome": "missing"})
            breakdown["missing"] += 1
            continue
        record = scoring[name]
        t1, t2, t3 = (record[k] for k in ("tier_1", "tier_2", "tier_3"))
        scores = [float(t["score"]) for t in (t1, t2, t3)]
        if not all(math.isfinite(n) for n in scores):
            raise ValueError(f"Non-finite score in {name}")
        message = str(t3.get("message", ""))
        if scores[0] != 1:
            outcome = "model_invalid"
        elif message == "Cable insertion successful." and scores[2] == 75:
            outcome = "full_insertion"
        elif "Incorrect Port" in message:
            outcome = "wrong_port"
        elif message.startswith("Partial insertion detected"):
            outcome = "partial_insertion"
        elif message.startswith("No insertion detected"):
            outcome = "proximity" if scores[2] > 0 else "no_insertion"
        else:
            outcome = "execution_or_scoring_failure"
        categories = t2.get("categories", {}) or {}
        penalties = {k: v for k, v in categories.items() if float(v["score"]) < 0}
        trials.append({"trial": name, "outcome": outcome, "total": sum(scores),
                       "tiers": record, "penalties": penalties})
        breakdown[outcome] += 1
    unexpected = sorted(set(scoring) - {"total"} - set(expected_trials))
    complete = not breakdown["missing"] and not unexpected
    if complete and not math.isclose(sum(t["total"] for t in trials), total, abs_tol=0.05):
        raise ValueError("Official total disagrees with per-trial scores")
    return {"complete": complete, "official_total": total, "trials": trials,
            "unexpected_trials": unexpected, "failure_breakdown": dict(breakdown),
            "full_insertion_rate": breakdown["full_insertion"] / len(expected_trials)}


def compose_config(run, evaluator, model, gpu=False, checkpoints=None, capture=True):
    def mount(source, target, read_only=False):
        return {"type": "bind", "source": str(run / source), "target": target, "read_only": read_only}
    credentials = {"AIC_ENABLE_ACL": "true", "AIC_MODEL_PASSWD": "isolated-baseline-model"}
    config = {
        "services": {
            "eval": {
                "image": evaluator,
                "command": ["gazebo_gui:=false", "launch_rviz:=false", "ground_truth:=false",
                            "start_aic_engine:=true", "shutdown_on_aic_engine_exit:=true",
                            "model_discovery_timeout_seconds:=30",
                            "aic_engine_config_file:=/benchmark/config.yaml"],
                "environment": {**credentials, "AIC_EVAL_PASSWD": "isolated-baseline-eval",
                                "AIC_RESULTS_DIR": "/results", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics,display",
                                "LIBGL_ALWAYS_SOFTWARE": "0" if gpu else "1"},
                "volumes": [mount("config.yaml", "/benchmark/config.yaml", True), mount("results", "/results")],
            },
            "model": {
                "image": model,
                "environment": {**credentials, "AIC_ROUTER_ADDR": "eval:7447",
                                "AIC_CAPTURE_DIR": "/captures",
                                "ZENOH_ROUTER_CHECK_ATTEMPTS": "-1"},
                "volumes": [mount("captures", "/captures")],
            },
        },
        "networks": {"default": {"internal": True}},
    }
    policy = config["services"]["model"]
    if not capture:
        # Capture writes compressed images inside the control loop; evaluation never does.
        del policy["environment"]["AIC_CAPTURE_DIR"]
        policy["volumes"] = []
    for kind, name in (checkpoints or {}).items():
        policy["environment"][CHECKPOINT_VARIABLES[kind]] = f"{CHECKPOINT_MOUNT}/{name}"
    if checkpoints:
        policy["volumes"].append({"type": "bind", "source": str(run / "checkpoints"),
                                  "target": CHECKPOINT_MOUNT, "read_only": True})
    if gpu:
        for service in config["services"].values():
            service["gpus"] = "all"
    return config


def build_verified_image(destination, manifest, tag_prefix, timeout, labels=()):
    """Build the frozen source and prove the installed policy matches it."""
    tag = tag_prefix + manifest['source_sha256'][:20]
    # The tag names the full source fingerprint, and installed code is verified
    # below, so an existing image is reused instead of re-resolving the base
    # image over the network. Registry timeouts are retried, then raised.
    reused = subprocess.run(['docker', 'image', 'inspect', tag], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0
    with (destination/'build.log').open('w') as log:
        for attempt in range(3 if not reused else 0):
            try:
                command(['docker', 'build', *[a for label in labels for a in ('--label', label)],
                         '-t', tag, '-f', 'docker/aic_model/Dockerfile', '.'],
                        cwd=destination/'source', stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                log.write(f'\nBuild attempt {attempt+1} failed; retrying\n'); log.flush()
                time.sleep(30*(attempt+1))
        if reused:
            log.write(f'Reused existing image {tag}; installed code verified below\n')
    info = image_info(tag)
    info['reused_existing_tag'] = reused
    installed = json.loads(probe_image(info['id'], ['run', '--as-is', 'python', '-c',
        "import hashlib,importlib.util,json,pathlib; "
        "p=pathlib.Path(importlib.util.find_spec('aic_model').origin).parent; "
        "print(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.py')}))"],
        stdout=subprocess.PIPE).stdout)
    expected = {Path(p).name: h for p,h in manifest['source_files'].items()
                if p.startswith('aic_model/aic_model/') and p.endswith('.py')}
    if installed != expected:
        raise ValueError('Installed policy differs from frozen source')
    return info


def probe_image(image, arguments, **kwargs):
    """Bounded offline inspection; also remove the container on client timeout."""
    name = "aic-baseline-probe-" + uuid.uuid4().hex[:12]
    try:
        return command(["docker", "run", "--rm", "--name", name, "--network", "none",
                        "--entrypoint", "/root/.pixi/bin/pixi", image, *arguments],
                       timeout=60, **kwargs)
    finally:
        subprocess.run(["docker", "rm", "--force", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)


def image_info(ref):
    info = json.loads(output(["docker", "image", "inspect", ref]))[0]
    return {"id": info["Id"], "repo_digests": info.get("RepoDigests", []), "reference": ref, "labels": info.get("Config", {}).get("Labels", {})}


def execute(run, evaluator, gpu, timeout):
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "prepared":
        raise ValueError("Prepare a new run; completed or failed directories are never reused")
    if sha256(__file__) != manifest["source_files"]["scripts/benchmark.py"]:
        raise ValueError("Runner changed since preparation; run the frozen source/scripts/benchmark.py or prepare again")
    compose = ["docker", "compose", "--project-name", "aic-baseline-" + uuid.uuid4().hex[:12],
               "-f", str(run / "compose.yaml")]
    compose_started = False
    try:
        current_files = {str(p.relative_to(run / "source")) for p in (run / "source").rglob("*") if p.is_file()}
        if current_files != set(manifest["source_files"]):
            raise ValueError("Frozen source file inventory changed")
        for path, digest in manifest["source_files"].items():
            if sha256(run / "source" / path) != digest:
                raise ValueError(f"Frozen source changed: {path}")
        if sha256(run / "config.yaml") != manifest["config_sha256"]:
            raise ValueError("Frozen configuration changed")
        manifest["status"] = "building"
        manifest["evaluator"] = image_info(evaluator)
        manifest["docker_version"] = output(["docker", "version", "--format", "{{.Server.Version}}"])
        manifest["gpu_requested"] = gpu
        write_json(manifest_path, manifest)
        manifest["model"] = build_verified_image(run, manifest, "aic-baseline-model:", timeout,
                                                 labels=("aic.source_sha256=" + manifest["source_sha256"],))
        with (run / "packages.json").open("w") as packages:
            probe_image(manifest["model"]["id"], ["list", "--frozen", "--no-install", "--json"], stdout=packages)
        manifest["installed_policy_verified"] = True
        probe_image(manifest["model"]["id"], ["run", "--as-is", "python", "-c",
            "from aic_model import sfp_face_decoder,sc_heatmap_detector,board_registration; "
            "from aic_model.vision_runtime import cv2; print('Vision imports OK:',cv2.__version__)"],
            stdout=subprocess.PIPE)
        manifest["vision_imports_verified"] = True
        if gpu:
            manifest["gpu"] = output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        (run / "compose.yaml").write_text(yaml.safe_dump(compose_config(
            run, manifest["evaluator"]["id"], manifest["model"]["id"], gpu,
            {kind: item["name"] for kind, item in manifest.get("checkpoints", {}).items()},
            manifest.get("capture_enabled", True))))
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        compose_started = True
        with (run / "compose.log").open("w") as log:
            command(compose + ["up", "--abort-on-container-exit", "--exit-code-from", "eval", "--pull", "never"],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        summary = summarize(yaml.safe_load((run / "results/scoring.yaml").read_text()), manifest["expected_trials"])
        write_json(run / "summary.json", summary)
        manifest["status"] = "completed" if summary["complete"] else "incomplete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if compose_started:
                with (run / "container-status.json").open("w") as status:
                    subprocess.run(compose + ["ps", "--all", "--format", "json"], stdout=status, timeout=30)
                command(compose + ["down", "--timeout", "10"], timeout=60)
        except Exception as cleanup_error:
            manifest["cleanup_error"] = str(cleanup_error)
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, default=ROOT / "benchmarks/qualification.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sc-checkpoint", type=Path, help="Frozen candidate mounted read-only in place of the image default")
    p.add_argument("--sfp-checkpoint", type=Path, help="Frozen candidate mounted read-only in place of the image default")
    p.add_argument("--plug-sfp-checkpoint", type=Path, help="Frozen plug-pose candidate (grasp measurement)")
    p.add_argument("--plug-sc-checkpoint", type=Path, help="Frozen plug-pose candidate (grasp measurement)")
    p.add_argument("--no-capture", action="store_true", help="Disable in-loop dataset capture (evaluation-like timing)")
    p = sub.add_parser("run")
    p.add_argument("directory", type=Path)
    p.add_argument("--eval-image", required=True, help="Locally available evaluator reference; resolved to immutable image ID")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--timeout", type=int, default=3600, help="Wall seconds per build/evaluation stage")
    p = sub.add_parser("summarize")
    p.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            checkpoints = {kind: path for kind, path in (("sc", args.sc_checkpoint), ("sfp", args.sfp_checkpoint),
                                                         ("plug_sfp", args.plug_sfp_checkpoint),
                                                         ("plug_sc", args.plug_sc_checkpoint))
                           if path is not None}
            print(prepare(args.config, args.output, checkpoints, not args.no_capture))
        elif args.action == "run":
            execute(args.directory.resolve(), args.eval_image, args.gpu, args.timeout)
        else:
            run = args.directory.resolve()
            manifest = json.loads((run / "manifest.json").read_text())
            summary = summarize(yaml.safe_load((run / "results/scoring.yaml").read_text()), manifest["expected_trials"])
            write_json(run / "summary.json", summary)
            print(json.dumps(summary, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
