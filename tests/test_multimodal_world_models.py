from dmxsf.multimodal.world_models import plan, traverse


def test_replanning_reaches_goal_after_repeat_in_trace():
    assert traverse(False)[-1] == (0, 1)
    trace = traverse(True)
    assert trace[-1] == (0, 3)
    assert len(trace) == 7 and trace[1] == trace[2]


def test_blocked_goal_has_no_path():
    assert plan((0, 0), (0, 3), {(0, 1), (1, 0)}, (2, 4)) is None
