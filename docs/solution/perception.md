# Perception: finding the target port

The policy has to find the requested port from the three wrist cameras alone.
It cannot read the task board's pose, the module placements or any other part
of the scene configuration. My approach anchors everything to the board first,
then looks for the target only where the public board geometry says it can be.

## 1. Register the board

The task board carries an asymmetric magenta marker. Its outline is known from
the public board asset (`Task Board Base/base_visual.glb`).
[`board_registration.py`](../../aic_model/aic_model/board_registration.py)
segments the marker in each camera image and fits it to that outline. Each
camera fit uses the robot's TF for the camera pose at exposure time, giving a
board pose in `base_link`.

- **Agreement.** Camera fits must agree with each other before they are fused.
  A single planar fit is not trusted on its own.
- **Level fit.** The board is level in every scene, because only its position
  and yaw are randomized. So after a free 6-DoF fit, I refit position and yaw
  with roll and pitch held at zero.
  - Free fits were tilted by up to 1.5°. On the far rails that moved the
    projected templates 2–5 mm across the rail, which the decoders cannot
    absorb.
  - A free fit tilted by more than 3° is rejected rather than levelled.
- **Partial marker.** When the marker is clipped at an image edge,
  [`partial_marker.py`](../../aic_model/aic_model/partial_marker.py) fits the
  visible part.
- **Marker search.** When no camera sees the marker from the start pose, the
  policy withdraws and pans to find it. It then retraces that search back to
  the start view, acting on target estimates it sees on the way.

## 2. Frame the requested rail

The task names the target module and port. Given the board pose and the public
rail bounds, [`rail_view.py`](../../aic_model/aic_model/rail_view.py) plans a
bounded camera motion so that the whole legal envelope of that rail fits in
the images:
- a withdrawal of at most 10 cm along the board normal;
- a wrist turn of at most 25°.

The motion holds as soon as the envelope is in view, stops on contact, and is
time-limited. Data collection shares the same planner, so training crops match
the views the policy sees at runtime.

## 3. Detect and decode the target

Both detectors are heatmap networks sharing one architecture
([`landmark_network.py`](../../aic_model/aic_model/landmark_network.py)). They
run on crops around the requested rail.

**SFP.** A NIC card carries two SFP ports 23.2 mm apart along the card's
sliding axis, while the card can slide much further than that, so one port
alone cannot tell them apart.
- The detector predicts the four corners and the centre of *both* port faces.
- [`sfp_face_decoder.py`](../../aic_model/aic_model/sfp_face_decoder.py) scores
  every legal card placement against the pooled corner heatmaps of both ports.
  The search covers translation over the mount's physical travel and yaw
  within ±10°.
- A placement shifted by one port pitch leaves a template port on empty
  support, so it scores low.
- The card geometry is in
  [`sfp_card_template.py`](../../aic_model/aic_model/sfp_card_template.py).

**SC.** An SC rail crop can contain both SC ports, so the detector is told
which rail to look at.
- The rail-conditioned input (`rail_conditioned_v1`) adds a fourth channel to
  the RGB crop: the projected envelope of the requested rail
  ([`rail_conditioning.py`](../../aic_model/aic_model/rail_conditioning.py)).
- [`sc_face_decoder.py`](../../aic_model/aic_model/sc_face_decoder.py) fits a
  face template along the rail.
- SC ports only translate along their rails, so the template yaw is bounded to
  ±2.5°: the public spec plus the measured registration error. The port's
  orientation comes from the registered board.

**Shared checks.** Both decoders reject a view when the best placement and the
best *distinct* competitor are too close in score. Distinct means more than
4 mm or 10° apart.

## 4. Fuse across cameras and lock

[`policy_perception.py`](../../aic_model/aic_model/policy_perception.py)
combines the accepted views, using the triangulation checks in
[`multiview.py`](../../aic_model/aic_model/multiview.py): positive depth,
parallax and reprojection error.
- **Triangulation.** The target's landmarks are triangulated from the
  accepted views and fitted rigidly to the public face model.
- **Agreement.** Views must agree on the placement they decoded. SC views must
  agree on the rail translation within 2 mm. With three cameras, the largest
  agreeing subset is kept.
- **Rail check.** The triangulated face must lie at a legal position for the
  requested port. This rejects a neighbouring module that happens to look
  right.
- **Lock.** A target is locked once recent estimates cluster: at least 10
  accepted estimates within the last 15 cycles, all within 5 mm of their
  median. Accepted SC poses flicker near the decoder margins, so demanding
  consecutive cycles only delayed locks.

On fresh scenes the locked estimates were typically within 0.5–1.6 mm of the
true port laterally. That is close to the insertion tolerance, and the face
search at the port covers the rest ([insertion.md](insertion.md)).

## 5. Measure the plug in the gripper

The policy aims the *plug tip*, not the TCP, so it needs the grasp.
- **Why the grasp varies.** Normally the grasp is fixed. But the evaluator can
  start a trial while the arm is still moving back from the previous one, and
  then the cable is attached with the plug shifted and turned in the gripper
  (see [insertion.md](insertion.md)).
- **How it is measured.**
  [`plug_pose.py`](../../aic_model/aic_model/plug_pose.py) takes a 384 px crop
  around the expected plug in each wrist camera and detects keypoints with a
  heatmap network.
  - SFP keypoints: the tip-face centre, its four corners and a point on the
    body axis.
  - SC keypoints: the two ferrule tips, the housing corners and an axis point.

  The keypoints are triangulated in the TCP frame, and a rigid fit gives the
  TCP-to-plug transform.
- **When it is used.** The measurement needs several still, consistent frames.
  It replaces the fixed grasp only when it differs by more than 2.5 mm or 3°.
  On shifted-grasp test scenes it matched ground truth to about 0.3 mm.

## What perception never uses

None of these steps reads ground truth, scene configuration or privileged
topics. The only inputs are camera images, camera info, the robot's TF and
joint state, and public asset geometry. Ground truth appears only in:
- training-data collection (labels);
- post-hoc evaluation.

[evaluation.md](evaluation.md) describes both.
