# Insertion: from a locked estimate to a seated plug

Perception gets the plug to within about a millimetre of the port. The
insertion tolerance is about the same: roughly 0.5 mm laterally and 1° of yaw
for SFP. So I don't try to make the estimate perfect. I approach compliantly,
let the plug land on the port face, and search there by contact.

The policy runs one control cycle at a time through explicit phases, with one
step method per phase in
[`policy_phases.py`](../../aic_model/aic_model/policy_phases.py):

```text
find_target → coarse_align → pre_insert → insert → settle
                    ↑______________ recover ______________|
```

## Before the task: a known start

The scoring evaluator resets the arm between trials. Its controller can
re-activate before the reset has taken effect. When that happens, a later
trial starts with the arm away from home: sometimes twisted, sometimes still
pressing on the board. The cable is then attached wherever the gripper happens
to be, so the plug sits shifted and turned in the gripper.

Before anything else, the policy:
1. **Returns to the start pose** (`Policy._restore_start_pose`).
   - It rises first, then moves and turns home at a bounded speed.
   - It stops on contact while rising from a calm start, or on a sustained
     load at home height.
   - It does nothing when the arm is already within 20 mm and 5° of home.
2. **Measures the grasp** with the wrist cameras ([perception.md](perception.md)).
3. **Samples a force baseline.** The wrist sensor reads about 21 N at rest,
   from the gripper and cable weight.

## Phases

- **`find_target`**: registers the board, frames the rail and waits for a
  locked target ([perception.md](perception.md)). A load from the
  freshly spawned cable pauses the framing motion instead of aborting it.
- **`coarse_align`**: moves to a standoff above the port and turns the plug
  onto the port axis. The orientation blends in over about 5 s. The port axis
  is smoothed, and flipped readings are rejected.
- **`pre_insert`**: centres the plug tip over the port at a short standoff.
  - The impedance stays deliberately soft (150 N/m). A stiff pre-insert
    centred the plug well, but the cable load then slid it off the face during
    the descent.
  - A proportional, capped correction feeds the measured plug-to-port residual
    back into the goal.
- **`insert`**: descends compliantly onto the face, then hands over to the
  face search.
- **`settle`**: confirms a deep insertion that happened without a face
  search, once the plug geometry is stable. A seat found by the face search
  ends the attempt directly from `insert`.
- **`recover`**: backs off and re-acquires. More than 5 retries ends the
  attempt.

### When the cable load wins

Under the cable's pull, the compliant arm can settle a few millimetres short
of the pre-insert goal, and no amount of centring moves it. Waiting out the
timeout only wastes time. Instead, pre-insert hands over to the face search
once the residual has stopped improving for a few seconds. The handover
limits are:
- **SFP:** 3.5 mm, the face search's spiral radius.
- **SC:** 6 mm, because the SC face search pulls the plug in from further off.

When the arm still swings off on landing after such a handover, the search
starts right away instead of recentring first.

## Face search at the port

[`face_search.py`](../../aic_model/aic_model/face_search.py) takes over once
the plug stops advancing near the estimated face. Stall detection uses position
only. A light press on the face barely changes the 21 N resting reading, so
force cannot detect face contact.

- **Search.** A constant-speed Archimedean spiral runs around the locked
  estimate: 0.6 mm pitch, 3.5 mm radius, 1.5 mm/s, about 43 s. At the same time
  the plug dithers ±1.2° in yaw at 1 Hz, pivoting about the plug tip.
  - Lateral stiffness is high (3000 N/m), so the tip tracks the spiral against
    friction.
  - The lateral target never leads the measured plug by more than 1.5 mm,
    which bounds the sideways load if the plug catches on a lead-in.
- **Entry.** Entry means advancing 1.5 mm past the stall point. Measuring from
  the stall cancels the estimate's depth error.
- **Seating.** After entry, the axial target leads the deepest tip position by
  6 mm, and the lateral target follows the plug.
  - A seat that stalls within 6 mm of the stall point was a false entry, and
    the search restarts there.
  - A seat that stalls short of a full advance gets a 6 s lateral wiggle. This
    is how SC plugs pass the adapter's detent 10.5 mm in; axial force alone
    did not. Full advance is 20 mm for SFP and 11.5 mm for SC. The wiggle
    does not always pass the detent: in two of three runs of images built
    from this repository, an SC plug that started normally stopped there and
    scored partial.
- **Dwell.** The port's touch sensor reports an insertion only after 1 s of
  continuous contact. So the seat is held for 2 s before the attempt ends,
  with a firmer press for SC.
- **Exhaustion.** An exhausted spiral recovers and re-acquires.

While a search is active, the force, orientation-drift and seating-timeout
aborts are suppressed. A plug dropping into the port legitimately twists the
TCP against its target.

## Ending an attempt

Every attempt ends the same way, including timeouts and failed acquisitions:
1. The policy parks the plug at the port entrance.
2. It logs `[attempt_end]` with the reason.
3. It returns `success=true`.

The evaluator then measures the outcome itself. A failed attempt can still earn
proximity points, but a policy's own success flag never counts as an
insertion.

## Force and contact limits

The evaluator penalizes forces above 20 N held for more than 1 s, and contact
with off-limit parts.
- Every motion runs under impedance control.
- Abort thresholds sit above the resting reading.
- Sustained-load checks mirror the evaluator's 1 s window.

## Known limitations

- **Shifted SFP grasps.** When the plug sits 42–46 mm out of the TCP (normally
  51–54 mm), a gripper finger touches the NIC card mount at full depth. The
  insertion then ends partial.
- **Shifted SC grasps** enter on the first search, but they rest 0.02–0.08 mm
  short of the port's contact box and score partial.
- **Thin SC stall margin.** Their pre-insert stall residual (about 4–6 mm) sits
  close to the 6 mm handover limit.
  - In one test, sending a command on the cycle that logs the 1 Hz SC
    pre-insert diagnostic, which normally skips its command, made the plug
    stall 0.5–1.5 mm further off. So I left the skip in place.
  - I don't understand that mechanism.
- **SFP stalls beyond 3.5 mm** under cable load keep re-centring until the
  timeout.
- **Start-pose return.** It can still be stopped by contact on the way back.
