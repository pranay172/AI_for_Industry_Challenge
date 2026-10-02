"""Face search: spiral coverage, tip pivot, entry and seating targets."""

import numpy as np

from aic_model import face_search as fs


def port_rotation(yaw_deg=10.):
    yaw = np.deg2rad(yaw_deg)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.])
    z = np.array([0., 0., -1.])
    return np.column_stack([x, np.cross(z, x), z])


def search(now=0.):
    tcp_rotation = fs.rotation([1., .3, .2], 2.4)          # tilted gripper
    tip_in_tcp = np.array([0., -.0207, .0541])
    port = np.array([-.42, .31, .18])
    plug = port+np.array([.0006, -.0004, .0013])           # stalled 1.3 mm above the face
    tcp = plug-tcp_rotation@tip_in_tcp
    return fs.start(port, port_rotation(), tcp, tcp_rotation, plug, now), tip_in_tcp


def test_spiral_grows_at_constant_pitch_and_is_spent_at_the_radius():
    radii = [np.linalg.norm(fs.spiral_offset(t)[0]) for t in np.linspace(0., fs.spiral_duration()*.99, 400)]
    assert np.all(np.diff(radii) >= -1e-12) and radii[0] == 0.
    assert radii[-1] <= fs.SPIRAL_MAX_RADIUS_M
    assert fs.spiral_offset(fs.spiral_duration()*1.01)[1]
    # Successive turns are one pitch apart.
    turn = lambda k: fs.SPIRAL_PITCH_M*k
    theta_k = lambda k: 2.*np.pi*k
    arc = lambda theta: fs.SPIRAL_PITCH_M*theta**2/(4.*np.pi)
    for k in (1, 2, 3):
        offset, _ = fs.spiral_offset(arc(theta_k(k))/fs.SPIRAL_SPEED_MPS)
        assert np.isclose(np.linalg.norm(offset), turn(k))


def test_spiral_speed_is_bounded():
    t = np.linspace(.5, fs.spiral_duration()*.99, 2000)
    points = np.array([fs.spiral_offset(v)[0] for v in t])
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1)/np.diff(t)
    assert speed.max() < 1.05*fs.SPIRAL_SPEED_MPS


def test_search_goal_turns_the_plug_about_its_tip_on_the_spiral():
    s, tip_in_tcp = search()
    assert np.isclose(s.stall_axial, -.0013)
    for now in (0.3, 2.7, 9.1):
        position, rotation = fs.search_goal(s, now)
        tip = position+rotation@tip_in_tcp
        local = s.port_rotation.T@(tip-s.port_position)
        assert np.allclose(local[:2], fs.spiral_offset(now)[0], atol=1e-12)
        assert np.isclose(local[2], s.stall_axial+fs.PRESS_DEPTH_M)
        turned = rotation@s.tcp_rotation.T
        angle = np.arctan2(np.dot(.5*np.array([turned[2, 1]-turned[1, 2], turned[0, 2]-turned[2, 0],
                                                turned[1, 0]-turned[0, 1]]), s.axis), .5*(np.trace(turned)-1.))
        assert np.isclose(angle, fs.yaw_offset(now))
    assert fs.search_goal(s, fs.spiral_duration()*1.01) is None


def test_search_target_leads_a_caught_plug_by_a_bounded_distance():
    s, tip_in_tcp = search()
    now = fs.spiral_duration()*.9                          # spiral near its full radius
    caught = s.port_position+s.axis*s.stall_axial          # plug held at the centre
    position, rotation = fs.search_goal(s, now, caught)
    local = s.port_rotation.T@(position+rotation@tip_in_tcp-s.port_position)
    assert np.isclose(np.linalg.norm(local[:2]), fs.MAX_LATERAL_LEAD_M)
    assert np.allclose(local[:2]/np.linalg.norm(local[:2]),
                       fs.spiral_offset(now)[0]/np.linalg.norm(fs.spiral_offset(now)[0]))
    # A plug that keeps up is not held back.
    on_path = at_depth(s, s.stall_axial, fs.spiral_offset(now)[0])
    local = s.port_rotation.T@(fs.search_goal(s, now, on_path)[0]+rotation@tip_in_tcp-s.port_position)
    assert np.allclose(local[:2], fs.spiral_offset(now)[0])


def test_entry_is_judged_from_the_stall_depth_not_the_estimated_face():
    s, _ = search()
    stalled = s.port_position+s.axis*s.stall_axial
    assert not fs.entered(s, stalled+s.axis*.001)
    assert fs.entered(s, stalled+s.axis*(fs.ENTRY_ADVANCE_M+.0002))


def at_depth(s, depth, lateral=(.0003, -.0002)):
    return s.port_position+s.port_rotation@np.array([lateral[0], lateral[1], depth])


def test_seating_keeps_the_entry_lateral_and_leads_the_deepest_tip():
    s, tip_in_tcp = search()
    fs.begin_seating(s, at_depth(s, .001), s.tcp_rotation, 10.)
    position, rotation = fs.seat_goal(s, 10.5)
    assert np.allclose(rotation, s.tcp_rotation)
    local = s.port_rotation.T@(position+rotation@tip_in_tcp-s.port_position)
    assert np.allclose(local[:2], [.0003, -.0002], atol=1e-12)
    assert np.isclose(local[2], .001+fs.SEAT_STEP_M)
    # Advancing moves the lateral target to where the port holds the plug.
    assert fs.seat_state(s, at_depth(s, .02, (-.0005, .0004)), 10.6) == "seating"
    local = s.port_rotation.T@(fs.seat_goal(s, 10.6)[0]+rotation@tip_in_tcp-s.port_position)
    assert np.allclose(local, [-.0005, .0004, .02+fs.SEAT_STEP_M])
    # A stalled plug that only rattles laterally does not move it.
    assert fs.seat_state(s, at_depth(s, .02, (.0005, .0004)), 10.7) == "seating"
    local = s.port_rotation.T@(fs.seat_goal(s, 10.7)[0]+rotation@tip_in_tcp-s.port_position)
    assert np.allclose(local[:2], [-.0005, .0004])


def test_a_seat_that_stalls_near_the_stall_point_was_no_entry():
    s, _ = search()
    fs.begin_seating(s, at_depth(s, s.stall_axial+.002), s.tcp_rotation, 10.)
    assert fs.seat_state(s, at_depth(s, s.stall_axial+.005), 10.5) == "seating"     # a 5 mm hover
    assert fs.seat_state(s, at_depth(s, s.stall_axial+.005), 10.5+fs.SEAT_STALL_SEC) == "false_entry"


def test_a_long_advance_is_done_once_it_stalls():
    s, _ = search()
    fs.begin_seating(s, at_depth(s, .002), s.tcp_rotation, 10.)
    assert fs.seat_state(s, at_depth(s, .04), 11.) == "seating"
    t = 11.+fs.SEAT_STALL_SEC
    assert fs.seat_state(s, at_depth(s, .04), t) == "dwell"
    assert fs.seat_state(s, at_depth(s, .04), t+fs.SEAT_DWELL_SEC-.1) == "dwell"
    assert fs.seat_state(s, at_depth(s, .04), t+fs.SEAT_DWELL_SEC) == "done"


def sc_search():
    s, tip_in_tcp = search()
    return fs.start(s.port_position, s.port_rotation, s.port_position, np.eye(3),
                    at_depth(s, .002), 0., mode="sc"), tip_in_tcp


def test_sc_full_seat_stops_but_the_detent_is_wiggled():
    # Stalled 2 mm inside the mouth: full seat (15.5 mm in) is done, the detent (10.5 mm) is not.
    s, _ = sc_search()
    fs.begin_seating(s, at_depth(s, .004), s.tcp_rotation, 10.)
    fs.seat_state(s, at_depth(s, .0155), 10.5)
    assert fs.seat_state(s, at_depth(s, .0155), 10.5+fs.SEAT_STALL_SEC) == "dwell"
    # The dwell presses the plug onto the port's contact sensor.
    force = np.array(fs.press_wrench_tcp(s, s.tcp_rotation)[:3])
    assert np.allclose(s.tcp_rotation@force, s.axis*fs.DWELL_FORCE_N["sc"])
    s, _ = sc_search()
    fs.begin_seating(s, at_depth(s, .004), s.tcp_rotation, 10.)
    fs.seat_state(s, at_depth(s, .0105), 10.5)
    assert fs.seat_state(s, at_depth(s, .0105), 10.5+fs.SEAT_STALL_SEC) == "wiggle"


def test_a_stalled_partial_seat_is_wiggled_then_released():
    s, tip_in_tcp = search()
    fs.begin_seating(s, at_depth(s, .002), s.tcp_rotation, 10.)
    assert fs.seat_state(s, at_depth(s, .0105), 11.) == "seating"      # SC detent
    t0 = 11.+fs.SEAT_STALL_SEC
    assert fs.seat_state(s, at_depth(s, .0105), t0) == "wiggle"
    assert fs.press_wrench_tcp(s, s.tcp_rotation) != fs.press_wrench_tcp(
        fs.FaceSearch(**{**s.__dict__, "wiggle_start": 0.}), s.tcp_rotation)
    offsets = []
    for t in np.linspace(t0, t0+fs.WIGGLE_PERIOD_SEC, 9):
        position, rotation = fs.seat_goal(s, t)
        assert np.allclose(rotation, s.tcp_rotation)                   # no yaw inside the port
        local = s.port_rotation.T@(position+rotation@tip_in_tcp-s.port_position)
        offsets.append(np.hypot(local[0]-.0003, local[1]+.0002))
    assert np.allclose(offsets, fs.WIGGLE_RADIUS_M)
    # Passing the detent resumes plain seating; a further stall wiggles again.
    assert fs.seat_state(s, at_depth(s, .0155), t0+1.) == "seating"
    assert fs.seat_state(s, at_depth(s, .0155), t0+1.+fs.SEAT_STALL_SEC) == "wiggle"
    t = t0+1.+fs.SEAT_STALL_SEC+fs.WIGGLE_SEC
    assert fs.seat_state(s, at_depth(s, .0155), t) == "dwell"
    assert fs.seat_goal(s, t)[1] is not None and s.wiggle_start == 0.      # hold still while dwelling
    assert fs.seat_state(s, at_depth(s, .0155), t+fs.SEAT_DWELL_SEC) == "done"


def test_press_wrench_pushes_along_the_insertion_axis_in_the_tcp_frame():
    s, _ = search()
    force = np.array(fs.press_wrench_tcp(s, s.tcp_rotation)[:3])
    assert np.allclose(s.tcp_rotation@force, s.axis*fs.SEARCH_FORCE_N)
    fs.begin_seating(s, s.port_position, s.tcp_rotation, 1.)
    force = np.array(fs.press_wrench_tcp(s, s.tcp_rotation)[:3])
    assert np.allclose(s.tcp_rotation@force, s.axis*fs.SEAT_FORCE_N)


def test_stall_counts_only_near_the_face():
    port, axis = np.zeros(3), np.array([0., 0., -1.])
    near = np.array([0., 0., .002])                  # 2 mm above the face
    assert fs.stalled_at_face(port, axis, near, fs.STALL_SEC)
    assert not fs.stalled_at_face(port, axis, near, fs.STALL_SEC-.1)
    assert not fs.stalled_at_face(port, axis, np.array([0., 0., .02]), fs.STALL_SEC)
    assert not fs.stalled_at_face(port, axis, np.array([0., 0., .004]), fs.STALL_SEC)   # 4 mm above
    assert fs.stalled_at_face(port, axis, np.array([0., 0., -.004]), fs.STALL_SEC)      # SC mouth
