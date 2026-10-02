"""ROS adapter for the bounded rail-view planner, shared with data collection."""
from copy import deepcopy
import numpy as np

from . import policy_geometry as geom
from . import policy_perception as perception
from .rail_view import plan_rail_view, rail_in_view, rail_distance_bound, RailDistanceError, TIMEOUT_SECONDS


def suspend_rail_view(policy, now):
    """Target evidence paused framing: stop the plan clock until framing resumes.

    Without this, a later resume would jump along the plan by the paused time
    and count it against the framing timeout.
    """
    if getattr(policy, '_rail_view_start', None) is not None and getattr(policy, '_rail_view_suspended', None) is None:
        policy._rail_view_suspended = now


def reset_rail_view(policy):
    """End this framing episode, e.g. at target lock; reacquisition plans afresh."""
    policy._rail_view_plan = None
    policy._rail_view_start = None
    policy._rail_view_suspended = None
    policy._rail_view_force_since = None


# The framing motion swings the freshly spawned cable: in the official sample's
# trial 2 the wrist read up to 6.5 N over its start value with no contact
# (three runs). A load above force_limit therefore pauses the plan and holds;
# only a sustained load, or one above abort_limit, ends the episode.
CONTACT_HOLD_SEC = 1.5


def rail_view_step(policy, parsed, task, now, force_limit, abort_limit=None):
    """Return (status, hold/move pose); failed statuses never issue commands.

    'rail_view_distance_limit' and 'rail_view_force_hold' return a hold pose:
    the caller keeps position.
    """
    abort_limit = force_limit if abort_limit is None else abort_limit
    if parsed.force_mag >= abort_limit:
        return 'rail_view_contact', None
    if parsed.force_mag >= force_limit:
        since = getattr(policy, '_rail_view_force_since', None)
        if since is None:
            since = policy._rail_view_force_since = now
        if now-since >= CONTACT_HOLD_SEC:
            return 'rail_view_contact', None
        suspend_rail_view(policy, now)
        return 'rail_view_force_hold', deepcopy(parsed.tcp_pose)
    policy._rail_view_force_since = None
    suspended = getattr(policy, '_rail_view_suspended', None)
    if suspended is not None:
        if policy._rail_view_start is not None:
            policy._rail_view_start += max(0., now-suspended)
        policy._rail_view_suspended = None
    start = getattr(policy, '_rail_view_start', None)
    views = {}
    for name in perception.synchronized_camera_names(policy, parsed):
        geometry = perception.camera_projection_matrix(policy, parsed.camera_info_map.get(name),
                                                       parsed, parsed.image_header_map[name])
        if geometry is not None:
            views[name] = geometry
    board = getattr(policy, '_board_pose', None)
    if board is None or 'center' not in views or len(views) < 2:
        if start is not None and now-start > TIMEOUT_SECONDS:
            return 'rail_view_timeout', None
        return 'waiting_for_camera_geometry', deepcopy(parsed.tcp_pose)
    tcp = parsed.tcp_pose
    translation = np.array([tcp.position.x,tcp.position.y,tcp.position.z])
    max_distance = float(policy.PLAUSIBLE_PORT_DISTANCE_MAX_M)
    try:
        if rail_distance_bound(board, task.target_module_name, translation) > max_distance:
            # The public envelope is conservative, and qualification places the
            # target in view at the start: keep perceiving here, never move
            # toward the board. The search timeout still bounds the episode.
            return 'rail_view_distance_limit', deepcopy(parsed.tcp_pose)
    except ValueError:
        return 'rail_view_invalid_geometry', None
    framed = sum(rail_in_view(board, task.target_module_name, parsed.image_map[name], geometry)
                 for name,geometry in views.items())
    if framed >= 2:
        # Framing motion is complete; only active framing counts toward timeout.
        suspend_rail_view(policy, now)
        return 'framed', deepcopy(parsed.tcp_pose)
    if start is not None and now-start > TIMEOUT_SECONDS:
        return 'rail_view_timeout', None
    if getattr(policy, '_rail_view_plan', None) is None:
        tcp = parsed.tcp_pose
        translation = np.array([tcp.position.x,tcp.position.y,tcp.position.z])
        try:
            policy._rail_view_plan = plan_rail_view(board,task.target_module_name,
                views['center'][1],views['center'][2],geom.quaternion_to_matrix(tcp.orientation),translation,
                max_tcp_distance=max_distance)
        except RailDistanceError:
            return 'rail_view_distance_limit', deepcopy(parsed.tcp_pose)
        except ValueError:
            return 'rail_view_invalid_geometry', None
        policy._rail_view_start = now
    rotation, translation = policy._rail_view_plan.pose_at(now-policy._rail_view_start)
    pose = deepcopy(parsed.tcp_pose)
    pose.position.x, pose.position.y, pose.position.z = map(float,translation)
    pose.orientation = geom.matrix_to_quaternion(rotation)
    return 'framing_requested_rail', pose
