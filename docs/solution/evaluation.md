# Training data, evaluation and results

## Ground truth: where it is and isn't used

The policy never reads ground truth, privileged topics or the scene
configuration at runtime. Ground truth is used only in three places, all
offline or opt-in:

- **Training collection.** `scripts/collect_initial_views.py` runs the policy's
  own view and rail-framing logic in simulation. `ground_truth.py` and
  `policy_capture.py` attach exposure-matched TF labels, but only when
  `AIC_CAPTURE_DIR` is set. The labels never steer the view or the crop.
- **Diagnostics.**
  - `ValidateAcquisition` measures acquisition error against ground truth.
  - `OracleGrasp` substitutes the true grasp, to measure what a perfect grasp
    measurement would be worth. It refuses to start unless
    `AIC_PRIVILEGED_DIAGNOSTIC=1` is set.
  - The submission image never selects either policy.
- **Post-hoc analysis.** `scripts/posthoc_bag.py` reads the evaluator's bags
  after a run, for example to see where the plug tip actually stopped.

`policy_capture.py` imports `ground_truth.py` only when a capture sample is
taken. `tests/test_baseline.py` checks that no runtime module names it and that
importing the policy loads none of the privileged modules.

## Data and training

1. **Scenes.** `scripts/generate_scenes.py` draws seeded scenes:
   - qualification-like scenes;
   - wide scenes covering the board poses of every organizer example and the
     full physical rail travel.

   `scripts/screen_scenes.py` keeps only scenes whose target port face is
   visible from the fixed start cameras. Qualification guarantees that, but
   moving a card to another rail can break it.

   For the grasp models, `scripts/make_grasp_scenes.py` perturbs the cable's
   spawn pose in the gripper, covering the shifted grasps seen after the
   evaluator's reset race. Its docstring has the exact commands.
2. **Captures.** `scripts/collect_initial_views.py --rail-views` captures the
   start views and the rail-framing views, one isolated simulator run per scene.
3. **Labels.**
   - SFP: `scripts/label_sfp_faces.py` labels both SFP port faces of the
     requested card, from the public card geometry and the target's exposure-time
     pose.
   - SC: `aic_model/tools/label_sc_port_dataset.py` labels the SC port face.
   - `scripts/prepare_rail_dataset.py` attaches the runtime-selected rail crop
     and envelope.
   - `scripts/build_wide_faces_dataset.py` runs these three steps for the wide
     captures, applies the train/validation plan and appends the result to the
     earlier training set.
   - `scripts/mask_gripper_landmarks.py` marks landmarks hidden behind the
     gripper as not visible, using a fixed per-camera mask
     (`scripts/data/gripper_masks.npz`).
4. **Training.**
   - `aic_model/tools/train_sc_port_detector.py` trains both port detectors
     (`--landmarks sfp_faces` for SFP).
   - `aic_model/tools/train_plug_pose.py` trains the grasp keypoint models.
   - Validation splits were fixed before training.
5. **Shipping.** `scripts/export_checkpoint.py` keeps only the keys the policy
   loads and tags each port detector with its template decoder. The weights
   are copied unchanged. Exporting the original training outputs reproduces
   the four shipped files byte for byte. Four checkpoints ship in
   `aic_model/models/`. `benchmarks/models.json` records each one's hash, size,
   training run and validation metrics. The benchmark runner refuses a
   checkpoint whose hash does not match.

| Checkpoint | Role |
|---|---|
| `sfp_port_detector.pt` | SFP port-face detector |
| `sc_port_detector.pt` | SC port detector, rail-conditioned |
| `sfp_plug_pose.pt` | SFP plug keypoints in the gripper |
| `sc_plug_pose.pt` | SC plug keypoints in the gripper |

The capture datasets themselves are not part of this repository.

## Benchmark runner

`scripts/benchmark.py` runs the official evaluator against a frozen copy of the
policy, so a result can be traced to exact inputs.
- **Isolation.** It uses an isolated Docker Compose project, with Zenoh access
  control on and ground-truth TF off.
- **Separation.** Only the evaluator container receives the scene
  configuration and the results mount.

```bash
.pixi/envs/default/bin/python scripts/benchmark.py prepare \
  --config benchmarks/qualification.yaml --output benchmark_runs/qualification-001

.pixi/envs/default/bin/python scripts/benchmark.py run \
  benchmark_runs/qualification-001 \
  --eval-image ghcr.io/intrinsic-dev/aic/aic_eval:latest --gpu

.pixi/envs/default/bin/python scripts/benchmark.py summarize benchmark_runs/qualification-001
```

- **`prepare`** freezes the Docker build inputs and records a hash for every
  file. That includes uncommitted source and checkpoint files.
- **`run`**:
  - resolves the evaluator to an immutable image ID;
  - builds the model from the frozen source with `pixi install --locked`;
  - checks that the installed policy matches the frozen source before
    launching.
- **Run directories.** Each holds:
  - `manifest.json`: inputs, image IDs, hardware mode and status;
  - the frozen `source/`;
  - build and compose logs;
  - the official `results/` and evaluator bags;
  - `summary.json`: per-trial official scores and outcomes.
- **Counting.** A policy's own success return is never counted as an
  insertion. Only the evaluator's outcome counts. Failed or incomplete runs
  are not counted as measurements.

`benchmarks/qualification.yaml` is the exact configuration the organizers used
for the qualification leaderboard. They released it after the deadline as
`aic_engine/config/eval_config.yaml` (toolkit commit 749f385), and
`qualification.source.json` records its source and hash. The "official
scenes" results below are on it.

## Demo recording

`scripts/demo/` records a run for the demo GIF without changing the
evaluator's launch, which keeps `gazebo_gui:=false`.

- **`record_run.sh`** starts `benchmark.py run`. Once the evaluator container
  is up, it attaches a view-only `gz sim -g` client to it. The client uses
  `demo_gui.config` (one 3D view) and renders to an Xvfb display.
- **`capture_frames.py`** saves that display at 8 fps.
- **`make_demo_gif.py`** cuts each trial from the evaluator's goal to its
  result, using the engine's log timestamps, and labels it with the official
  outcome.

The viewer lowers the simulation's real-time factor (about 0.5–0.6, against
0.6–0.8 without it), so recorded runs are not quite evaluation conditions.
`docs/assets/demo.gif` comes from the second of two recorded runs, the first
in which all three trials were full insertions:

```bash
scripts/demo/make_demo_gif.py benchmark_runs/eval-demo-002 docs/assets/demo.gif \
  --speed 4 --fps 10 --width 720 --crop 60 0 960 430
```

## README figures

`scripts/figures/` draws the README's figures. The charts are SVG, written
twice for GitHub's light and dark themes.

| Figure | Script | Source |
|---|---|---|
| `banner.jpg` | `banner.py` | A frame of the recorded demo run, laid out and rendered with headless Chrome |
| `heldout-*.svg` | `results_chart.py` | `benchmarks/heldout_results.json`, the per-trial official totals of the held-out run |
| `pipeline-*.svg` | `pipeline_figure.py` | Drawn from the stage list in the script |
| `perception.png` | `perception_figure.py` | The shipped detectors replayed on wrist-camera exposures captured during a run (no ground truth) |
| `insertion-*.svg` | `insertion_trace.py`, `insertion_figure.py` | The plug and port poses from the evaluator's bag (ground truth, read after the run) and the phase boundaries from the policy's log |

```bash
cd scripts/figures
python banner.py ../../benchmark_runs/eval-demo-002/demo_frames/001020.jpg ../../docs/assets/banner.jpg
python results_chart.py ../../benchmarks/heldout_results.json ../../docs/assets/heldout
python pipeline_figure.py ../../docs/assets/pipeline
python perception_figure.py ../../benchmark_runs/publish-check-001/captures ../../docs/assets/perception.png \
  --sfp 261269f9f8d84ce7af300e724c1a161f_22456_000030.json \
  --sc a9f3f5ae76984ef7914e836ae9e68423_64209_000002.json
python insertion_figure.py trial1.csv ../../benchmark_runs/eval-demo-002/compose.log --trial 1 \
  ../../docs/assets/insertion
```

`insertion_trace.py` writes `trial1.csv` and runs inside the evaluation image,
which has the bag reader; its docstring has the command. The run directories
are local and not part of this repository.

## Tests

```bash
pixi run python -m pytest tests
```

The offline suite (about 250 tests, also run by `.github/workflows/tests.yml`)
covers:
- result classification and evaluator/policy separation;
- board registration, decoders, triangulation and grasp measurement;
- the pre-insert handover, the face search, the start-pose return and the
  lifecycle edge cases.

An optional ROS integration test checks real lifecycle and action transitions
against an isolated Zenoh router:

```bash
bash scripts/test_ros_lifecycle.sh
```

## Results

All scores are official evaluator totals. A trial is worth up to 100: Tier 3
gives 75 for a full insertion, up to 50 for a partial one and up to 25 for
proximity.

**The final code** (the submission image, and images built from this
repository):

| Set | Result |
|---|---|
| Submission image, `qualification.yaml`, 3 runs | 286.0 / 190.1 / 255.7 (mean 244.0); 7/9 full |
| Images built from this repository, `qualification.yaml`, 3 runs | 284.7 (3/3 full, the demo GIF), 255.3 and 229.5 (2/3 full each) |

**Earlier development revisions:**

| Set | Result |
|---|---|
| Held-out: 15 scenes, pre-registered, run once | **1109.2 / 1500**, 11/15 full insertions |
| `qualification.yaml`, 5 runs without an evaluator reset race | 276.7–286.2 per run, 3/3 full each |
| `sample_config.yaml` | 282.0 / 300, 3/3 full |
| Wide board-pose scenes (22) | 1450.0 / 2200, 14/22 full; target acquired in 20/22 (separate acquisition check) |

**How to read these:**
- The official scenes were used during development, so their scores overstate
  performance.
- The held-out set is the unbiased estimate, but of an earlier revision. Its
  seeds were fixed before the run, it was run once, and nothing was changed in
  response. That revision predates the wider SFP card search and the
  retrained detectors, the pre-insert plateau handover, the start-pose return
  and the grasp measurement. The set is spent, so the final code has no
  held-out score.
- In the submission-image check, every trial that started from the home pose
  was a full insertion. Both losses came after the evaluator reset race.
  - In one, the start-pose return was stopped by contact.
  - In the other, a shifted SFP grasp seated partially.
- Images built from this repository have the same weights as the submission
  image. Their policy code differs in three ways:
  - removed dead code and test-only helpers, which does not change behaviour;
  - a fix to the counter that re-seeds the smoothed port axis, a re-seed that
    never triggered in any recorded run;
  - the capture-only ground-truth module is imported only when capturing.

  They were run three times on `qualification.yaml`, none of them under exact
  evaluation conditions:

  | Run | Conditions | Total | Trials (SFP, SFP, SC) |
  |---|---|---|---|
  | Publish check | in-loop data capture on | 255.3 | 95.0, 94.9, 65.4 (partial) |
  | Recorded run 1 | demo viewer attached | 229.5 | 86.9, 96.2, 46.4 (partial) |
  | Recorded run 2 | demo viewer attached | **284.7** | 94.2, 94.6, 95.8 |

  - **SC partials.** Both started normally, found the opening and stopped at
    the port's detent, 9–10.5 mm in.
  - **Run 1, trial 1.** A false entry made the face search start again, so the
    trial took 83 s instead of about 24 s.
  - **The demo.** The GIF is recorded run 2, the best of the two recorded
    runs.
