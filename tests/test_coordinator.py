"""Round-loop tests: whole rounds run in milliseconds using a fake clock and a fake evaluator."""
import pytest

from server.coordinator import Coordinator, JobConflict
from shared.models import get_spec
from shared.serialization import bytes_to_state_dict


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeEvaluator:
    """Returns accuracies from a list, one per call (the last value repeats)."""

    def __init__(self, accuracies):
        self.accuracies = list(accuracies)
        self.calls = []

    def __call__(self, model_name, state_dict, full, **kwargs):
        self.calls.append(full)
        return self.accuracies.pop(0) if len(self.accuracies) > 1 else self.accuracies[0]


def make_coord(accuracies=(0.5,), min_nodes=2):
    clock = FakeClock()
    coord = Coordinator(FakeEvaluator(accuracies), heartbeat_timeout=20, round_deadline=180,
                        min_nodes=min_nodes, checkpoint_dir=None, clock=clock)
    return coord, clock


def start_job(coord, target=0.8, max_rounds=5):
    return coord.create_job("test-job", "cifar10", "cnn_small", target, max_rounds, 1)


def trained_weights(coord):
    """Pretend to train: download the current weights and send them back unchanged."""
    data, _, _ = coord.get_weights()
    return bytes_to_state_dict(data)


def report_all(coord, node_ids):
    for nid in node_ids:
        task = coord.get_task(nid)
        assert task["status"] == "train"
        assert coord.submit_update(nid, task["round"], trained_weights(coord)) == "accepted"


def test_waits_until_enough_nodes():
    coord, _ = make_coord(min_nodes=2)
    job = start_job(coord)
    coord.register("a", 4, "cpu")
    coord.tick()
    assert coord.round is None and job.status == "waiting_for_nodes"
    coord.register("b", 4, "cpu")
    coord.tick()
    assert coord.round is not None and job.status == "training"


def test_each_node_gets_a_different_slice():
    coord, _ = make_coord()
    start_job(coord)
    for nid in "abc":
        coord.register(nid, 4, "cpu")
    coord.tick()
    slices = {coord.get_task(nid)["slice_id"] for nid in "abc"}
    assert slices == {0, 1, 2}
    assert all(coord.get_task(nid)["num_slices"] == 3 for nid in "abc")


def test_unknown_node_gets_none_and_idle_node_waits():
    coord, _ = make_coord()
    assert coord.get_task("ghost") is None
    coord.register("a", 4, "cpu")
    assert coord.get_task("a") == {"status": "wait"}


def test_full_round_records_history():
    coord, clock = make_coord(accuracies=[0.42])
    job = start_job(coord)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    clock.advance(30)
    report_all(coord, ["a", "b"])
    coord.tick()                                     # closes, aggregates, evaluates
    assert coord.round is None
    assert job.current_round == 1
    assert job.history[0]["reported"] == 2 and job.history[0]["dropped"] == 0
    assert job.history[0]["accuracy"] == pytest.approx(0.42)


def test_server_counts_samples_itself():
    coord, _ = make_coord()
    start_job(coord)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    task = coord.get_task("a")
    coord.submit_update("a", task["round"], trained_weights(coord), claimed_samples=999_999)
    _, n = coord.round.updates["a"]
    assert n == 25_000                               # half of CIFAR-10, not the claim


def test_stale_unselected_and_duplicate_updates_are_rejected():
    coord, _ = make_coord()
    start_job(coord)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    weights = trained_weights(coord)
    coord.register("late", 4, "cpu")                 # joined after the round opened
    assert coord.submit_update("a", 99, weights) == "not_open"
    assert coord.submit_update("late", 1, weights) == "not_selected"
    assert coord.submit_update("a", 1, weights) == "accepted"
    assert coord.submit_update("a", 1, weights) == "duplicate"


def test_dropped_node_does_not_block_the_round():
    coord, clock = make_coord()
    job = start_job(coord)
    for nid in "abc":
        coord.register(nid, 4, "cpu")
    coord.tick()
    report_all(coord, ["a", "b"])                    # c never reports...
    clock.advance(25)                                # ...and stops heartbeating
    coord.heartbeat("a")
    coord.heartbeat("b")
    coord.tick()
    assert coord.round is None
    assert job.history[0]["reported"] == 2 and job.history[0]["dropped"] == 1
    assert coord.nodes["c"].reliability == 0.0


def test_straggler_is_cut_off_at_the_deadline():
    coord, clock = make_coord()
    job = start_job(coord)
    for nid in "abc":
        coord.register(nid, 4, "cpu")
    coord.tick()
    report_all(coord, ["a", "b"])
    for _ in range(10):                              # c stays alive but never finishes
        clock.advance(19)
        for nid in "abc":
            coord.heartbeat(nid)
        coord.tick()
    assert job.current_round == 1 and job.history[0]["dropped"] == 1


def test_round_with_no_updates_is_retried():
    coord, clock = make_coord()
    job = start_job(coord)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    clock.advance(200)                               # everyone vanished
    coord.tick()
    assert job.current_round == 0 and not job.history


def test_reaching_the_target_finishes_the_job_after_a_full_check():
    coord, _ = make_coord(accuracies=[0.85, 0.84])   # subset says 85%, full test confirms 84%
    job = start_job(coord, target=0.8)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    report_all(coord, ["a", "b"])
    coord.tick()
    assert job.status == "done"
    assert coord.evaluate_fn.calls == [False, True]
    assert coord.job_model(job.job_id) is not None


def test_round_limit_stops_the_job():
    coord, _ = make_coord(accuracies=[0.3])
    job = start_job(coord, target=0.9, max_rounds=2)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    for _ in range(2):
        coord.tick()
        report_all(coord, ["a", "b"])
        coord.tick()
    assert job.status == "stopped" and job.current_round == 2


def test_standby_nodes_are_not_selected():
    coord, _ = make_coord(min_nodes=1)
    start_job(coord)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu", available=False)
    coord.tick()
    assert set(coord.round.assignments) == {"a"}


def test_only_one_job_at_a_time():
    coord, _ = make_coord()
    start_job(coord)
    with pytest.raises(JobConflict):
        start_job(coord)


def test_unknown_model_is_rejected():
    coord, _ = make_coord()
    with pytest.raises(ValueError):
        coord.create_job("x", "cifar10", "no_such_model", 0.8, 5, 1)


def test_weights_must_match_the_jobs_model():
    coord, _ = make_coord()
    start_job(coord)                                 # cnn_small
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    wrong = get_spec("cnn_bn").make_model().state_dict()
    assert coord.check_update_shape(wrong) is not None
    assert coord.check_update_shape(trained_weights(coord)) is None


def test_reregistering_keeps_history():
    coord, _ = make_coord()
    start_job(coord)
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    report_all(coord, ["a", "b"])
    coord.tick()
    coord.register("a", 4, "cpu")                    # e.g. after a client restart
    assert coord.nodes["a"].rounds_completed == 1
    
def test_volunteers_keep_their_slice_between_rounds():
    coord, clock = make_coord()
    start_job(coord)
    for nid in "abc":
        coord.register(nid, 4, "cpu")
    coord.tick()
    first = dict(coord.round.assignments)
    clock.advance(5)
    for nid in "cab":                                # report in a different order...
        task = coord.get_task(nid)
        coord.submit_update(nid, task["round"], trained_weights(coord))
        clock.advance(3)                             # ...so their measured speeds differ
    coord.tick()                                     # close round 1
    coord.tick()                                     # open round 2
    assert coord.round.number == 2
    assert coord.round.assignments == first          # nobody has to fetch new data


class FakeStore:
    """Stands in for server.datasets.DatasetStore: one 2-class dataset of 100 training images."""
    def meta(self, dataset_id):
        if dataset_id != "pets-1":
            raise ValueError(f"no dataset '{dataset_id}'")
        return {"id": "pets-1", "name": "Pets", "num_classes": 2, "train_count": 100}


def test_job_on_a_stored_dataset():
    clock = FakeClock()
    coord = Coordinator(FakeEvaluator([0.6]), checkpoint_dir=None, clock=clock, dataset_store=FakeStore())
    job = coord.create_job("pets", "pets-1", "cnn_small", 0.9, 3, 1)
    assert job.num_classes == 2 and job.dataset_name == "Pets"
    final_layer = bytes_to_state_dict(job.weights)["layers.12.bias"]
    assert final_layer.shape == (2,)                 # model built with 2 outputs
    coord.register("a", 4, "cpu")
    coord.register("b", 4, "cpu")
    coord.tick()
    task = coord.get_task("a")
    assert task["dataset"] == "pets-1" and task["num_classes"] == 2
    coord.submit_update("a", task["round"], trained_weights(coord))
    _, n = coord.round.updates["a"]
    assert n == 50                                   # half of this dataset's 100 images, not of CIFAR-10
    with pytest.raises(ValueError):
        coord.create_job("x", "no-such-dataset", "cnn_small", 0.9, 3, 1)