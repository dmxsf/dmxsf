from dmxsf.post_training.rollout_pipeline import Trace, simulate


def test_publication_is_visible_when_actor_starts_at_same_time():
    assert simulate([2, 2, 2], 2, overlap=True, max_lag=1) == Trace(8, 3, 0, (0, 1, 1))


def test_filtered_batches_do_not_publish_weights():
    assert simulate([3, 3, 9, 2, 2, 2], 6, overlap=True, max_lag=1) == Trace(
        29, 4, 2, (0, 1, 2, 0, 1, 2)
    )
