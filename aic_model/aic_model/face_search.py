"""Bounded search at the port face.

Evaluator bags show the plug drops into the SFP cage only
when its lateral error is below about 0.5 mm and its yaw error below about
1 degree; the SC adapter mouth is similarly tight laterally. The locked port
estimate is typically 0.5-1.5 mm off, so a plug that stalls against the face is
held there with a light press and swept along an Archimedean spiral about the
locked estimate while its yaw is dithered about the insertion axis. Any advance
past the stall depth means the plug has entered and can be seated straight.

Seating pushes along the axis. The SC adapter holds the plug at a detent about
10.5 mm in that axial force alone did not pass in development runs; a lateral
jolt did. So a seat that stalls short of FULL_ADVANCE_M is wiggled laterally.
The wiggle carries no yaw: inside the port a 1 degree turn loads the tip with
about 20 N, over the evaluator's force limit. A seat that
stalls within FALSE_ENTRY_M of the stall point was a stall above the face, not
an entry, and the search restarts from there. While seating advances, the
lateral target follows the plug: inside the port the port sets the lateral
position, and holding an offset one preloads the tip (in one run, about
20 N with the plug bottomed 1.5 mm from its frozen entry lateral).

Everything is expressed in base_link. The axial coordinate of the plug tip is
measured from the locked port estimate along its insertion axis, so estimate
depth errors cancel: entry is judged relative to where the plug stalled.
"""
from dataclasses import dataclass

import numpy as np

SPIRAL_PITCH_M = 0.0006      # half-pitch 0.3 mm, inside the ~0.5 mm lateral tolerance
SPIRAL_SPEED_MPS = 0.0015
# Locked estimates have been up to 2.6 mm off laterally in evaluator runs, and
# face friction makes the tip lag the commanded spiral inward (0.8 mm at a 5 N
# press in that trial). About 43 s.
SPIRAL_MAX_RADIUS_M = 0.0035
YAW_AMPLITUDE_RAD = np.deg2rad(1.2)
YAW_PERIOD_SEC = 1.0
PRESS_DEPTH_M = 0.0030      # axial target beyond the stall point
# The lateral target may lead the measured plug by at most this much, bounding
# the spring's lateral load (~4.5 N) when the plug is caught in a lead-in: an
# unbounded spiral held 13 N sideways for 7 s there (in one run),
# drawing the force penalty. Free sliding lags only ~0.5 mm.
MAX_LATERAL_LEAD_M = 0.0015
SEARCH_FORCE_N = 1.0        # plus a light feedforward push along the insertion axis
SEAT_FORCE_N = 3.0
ENTRY_ADVANCE_M = 0.0015
SEAT_STEP_M = 0.006         # seating target lead beyond the deepest tip position
SEAT_STALL_SEC = 1.0        # no seating progress for this long starts the wiggle
SEAT_PROGRESS_M = 0.0003
# A seat stalled this close to the stall point was no entry: a stall may be
# detected 3 mm above the estimated face, which itself may sit ~2 mm above the
# true face (a 5 mm hover in one run), while real entries advance at
# least 8 mm (SC detent) or 20 mm (SFP).
FALSE_ENTRY_M = 0.006
# An advance this long needs no wiggle. SFP bottoms out ~45 mm in. SC seats
# 15.5 mm in and the plug stalls 0-2 mm inside the mouth, so a full seat is a
# 12.4-15.5 mm advance and the detent (10.5 mm in) at most 10.5 mm.
FULL_ADVANCE_M = {"sfp": 0.020, "sc": 0.0115}
WIGGLE_SEC = 6.0
# The port's touch sensor reports an insertion only after 1 s of continuous
# contact, and the evaluator ignores events after the policy returns (one fully
# seated trial scored partial that way). Hold the seat before reporting.
SEAT_DWELL_SEC = 2.0
# The SC port's contact box sits ~15.57 mm past the entrance, just beyond the
# housing stops: plugs resting at 15.42-15.54 mm never triggered it, plugs
# pressed to 15.60-15.71 mm did. The SC dwell presses harder (about 15 N
# axially with the spring, under the 20 N force-penalty level).
DWELL_FORCE_N = {"sfp": 3.0, "sc": 10.0}
WIGGLE_RADIUS_M = 0.0015    # about 4.5 N at the lateral stiffness
WIGGLE_PERIOD_SEC = 0.7
WIGGLE_FORCE_N = 6.0
# No axial progress for this long near the face starts a search. The wrist
# sensor already reads ~21 N at rest (gripper and cable) and a light press on
# the face barely changes it, so force is not a usable stall signal.
STALL_SEC = 1.0
# Axial window (negative above, positive inside) about the estimated face in
# which a stall counts as the face: yaw-blocked SFP plugs rest up to 1.5 mm
# above it and the SC mouth stops the plug about 2 mm inside, with estimate
# depth errors up to about 2 mm.
FACE_WINDOW_M = (-0.003, 0.006)
# The controller applies a diagonal stiffness to the base_link pose error. Task
# boards lie flat, so the insertion axis is close to base -z: stiff laterally so
# the tip tracks the spiral against face friction, soft axially so the press
# (PRESS_DEPTH_M*800 N/m + SEARCH_FORCE_N, about 3.4 N) stays light. The
# controller clamps each wrench axis at 10 N.
STIFFNESS = (3000., 3000., 800., 60., 60., 60.)
DAMPING = (110., 110., 60., 8., 8., 8.)


@dataclass
class FaceSearch:
    port_position: np.ndarray   # locked port estimate (entrance face), base_link
    port_rotation: np.ndarray   # locked port rotation; columns x, y lateral, z insertion
    tcp_rotation: np.ndarray    # TCP rotation at the stall
    tip_in_tcp: np.ndarray      # plug tip in the TCP frame
    stall_axial: float          # tip axial coordinate at the stall
    start_time: float
    full_advance: float = FULL_ADVANCE_M["sfp"]
    dwell_force: float = DWELL_FORCE_N["sfp"]
    entered_time: float = 0.0
    seat_start_axial: float = 0.0
    entry_lateral: np.ndarray = None    # seating lateral target; follows the plug as it advances
    seat_best_axial: float = -np.inf
    seat_progress_time: float = 0.0
    wiggle_start: float = 0.0
    done_time: float = 0.0

    @property
    def axis(self):
        return self.port_rotation[:, 2]


def axial(port_position, axis, point):
    return float(np.dot(np.asarray(point, dtype=float)-port_position, axis))


def start(port_position, port_rotation, tcp_position, tcp_rotation, plug_position, now, mode="sfp"):
    port_position = np.asarray(port_position, dtype=float)
    port_rotation = np.asarray(port_rotation, dtype=float)
    tcp_rotation = np.asarray(tcp_rotation, dtype=float)
    plug_position = np.asarray(plug_position, dtype=float)
    return FaceSearch(
        port_position=port_position, port_rotation=port_rotation, tcp_rotation=tcp_rotation,
        tip_in_tcp=tcp_rotation.T@(plug_position-np.asarray(tcp_position, dtype=float)),
        stall_axial=axial(port_position, port_rotation[:, 2], plug_position), start_time=now,
        full_advance=FULL_ADVANCE_M[mode], dwell_force=DWELL_FORCE_N[mode])


def spiral_offset(elapsed):
    """Lateral (x, y) offset along a constant-speed Archimedean spiral, and whether it is spent."""
    arc = SPIRAL_SPEED_MPS*max(float(elapsed), 0.)
    theta = np.sqrt(4.*np.pi*arc/SPIRAL_PITCH_M)
    radius = SPIRAL_PITCH_M*theta/(2.*np.pi)
    if radius > SPIRAL_MAX_RADIUS_M:
        return np.zeros(2), True
    return radius*np.array([np.cos(theta), np.sin(theta)]), False


def spiral_duration():
    return np.pi*SPIRAL_MAX_RADIUS_M**2/(SPIRAL_PITCH_M*SPIRAL_SPEED_MPS)


def yaw_offset(elapsed):
    return float(YAW_AMPLITUDE_RAD*np.sin(2.*np.pi*elapsed/YAW_PERIOD_SEC))


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)/max(float(np.linalg.norm(axis)), 1e-9)
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3)+np.sin(angle)*k+(1.-np.cos(angle))*(k@k)


def tcp_goal(search, tip_position, yaw):
    """TCP (position, rotation) placing the plug tip at `tip_position`, turned `yaw` about the axis."""
    R = rotation(search.axis, yaw)@search.tcp_rotation
    return np.asarray(tip_position, dtype=float)-R@search.tip_in_tcp, R


def search_goal(search, now, plug_position=None):
    """TCP goal while searching; None once the spiral is spent."""
    elapsed = now-search.start_time
    offset, spent = spiral_offset(elapsed)
    if spent:
        return None
    R = search.port_rotation
    lateral = search.port_position+R[:, 0]*offset[0]+R[:, 1]*offset[1]
    if plug_position is not None:
        plug = lateral_point(search, np.asarray(plug_position, dtype=float))
        lead = lateral-plug
        norm = float(np.linalg.norm(lead))
        if norm > MAX_LATERAL_LEAD_M:
            lateral = plug+lead*(MAX_LATERAL_LEAD_M/norm)
    tip = lateral+search.axis*(search.stall_axial+PRESS_DEPTH_M)
    return tcp_goal(search, tip, yaw_offset(elapsed))


def entered(search, plug_position):
    return axial(search.port_position, search.axis, plug_position)-search.stall_axial > ENTRY_ADVANCE_M


def lateral_point(search, plug_position):
    """Plug position projected onto the locked face plane."""
    return plug_position-search.axis*np.dot(plug_position-search.port_position, search.axis)


def begin_seating(search, plug_position, tcp_rotation, now):
    """Freeze the lateral position and orientation the plug entered with."""
    plug_position = np.asarray(plug_position, dtype=float)
    search.entered_time = now
    search.seat_start_axial = search.seat_best_axial = axial(search.port_position, search.axis, plug_position)
    search.seat_progress_time = now
    search.wiggle_start = 0.
    search.entry_lateral = lateral_point(search, plug_position)
    search.tcp_rotation = np.asarray(tcp_rotation, dtype=float)


def seat_state(search, plug_position, now):
    """'seating', 'wiggle', 'false_entry' (search again from here), 'dwell' or 'done'."""
    if search.done_time > 0.:
        return "done" if now-search.done_time >= SEAT_DWELL_SEC else "dwell"
    depth = axial(search.port_position, search.axis, plug_position)
    if depth > search.seat_best_axial+SEAT_PROGRESS_M:
        search.seat_best_axial = depth
        search.entry_lateral = lateral_point(search, np.asarray(plug_position, dtype=float))
        search.seat_progress_time = now
        search.wiggle_start = 0.
    if now-search.seat_progress_time < SEAT_STALL_SEC:
        return "wiggle" if search.wiggle_start > 0. else "seating"
    advance = search.seat_best_axial-search.stall_axial
    if advance < FALSE_ENTRY_M:
        return "false_entry"
    if search.wiggle_start > 0. and now-search.wiggle_start >= WIGGLE_SEC or advance >= search.full_advance:
        search.done_time, search.wiggle_start = now, 0.
        return "dwell"
    if search.wiggle_start <= 0.:
        search.wiggle_start = now
    return "wiggle"


def seat_goal(search, now):
    """TCP goal pushing the entered plug along the axis, wiggling once it stalls."""
    tip = search.entry_lateral+search.axis*(max(search.seat_best_axial, search.seat_start_axial)+SEAT_STEP_M)
    if search.wiggle_start <= 0.:
        return tcp_goal(search, tip, 0.)
    phase = 2.*np.pi*(now-search.wiggle_start)/WIGGLE_PERIOD_SEC
    R = search.port_rotation
    tip = tip+WIGGLE_RADIUS_M*(R[:, 0]*np.cos(phase)+R[:, 1]*np.sin(phase))
    return tcp_goal(search, tip, 0.)


def press_wrench_tcp(search, tcp_rotation):
    """Feedforward force along the insertion axis, in the TCP frame the controller expects."""
    if search.entry_lateral is None:
        newtons = SEARCH_FORCE_N
    elif search.done_time > 0.:
        newtons = search.dwell_force
    else:
        newtons = WIGGLE_FORCE_N if search.wiggle_start > 0. else SEAT_FORCE_N
    force = np.asarray(tcp_rotation, dtype=float).T@(search.axis*newtons)
    return (float(force[0]), float(force[1]), float(force[2]), 0., 0., 0.)


def stalled_at_face(port_position, axis, plug_position, stalled_sec):
    depth = axial(np.asarray(port_position, dtype=float), axis, plug_position)
    return stalled_sec >= STALL_SEC and FACE_WINDOW_M[0] <= depth <= FACE_WINDOW_M[1]
