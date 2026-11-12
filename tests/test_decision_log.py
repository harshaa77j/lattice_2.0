import json
from rl.decision_log import build_record, trace_record, append_jsonl

def test_build_record():
    nodes = [
        {"node_id": "A"},
        {"node_id": "B"},
        {"node_id": "C"},
    ]
    scores = {"A": 0.9, "B": 0.3, "C": 0.7}
    chosen = ["A", "C"]

    record = build_record(1, {"name": "test-job"}, nodes, scores, chosen)

    assert record["scheduler"] == "rl"
    assert record["round"] == 1
    assert len(record["candidates"]) == 3

    by_id = {n["node_id"]: n for n in record["candidates"]}
    assert by_id["A"]["chosen"] is True
    assert by_id["A"]["order"] == 0
    assert by_id["B"]["chosen"] is False
    assert by_id["B"]["order"] is None
    assert by_id["C"]["order"] == 1

def test_trace_record():
    record = trace_record(
        2, {"name": "test-job"}, ["A", "B", "C"],
        ["A", "B"], ["A"], 12.5, "rl", 0.75
    )
    assert record["chosen"] == ["A", "B"]
    assert record["reported"] == ["A"]
    assert record["dropped"] == ["B"]
    assert record["round_time_s"] == 12.5

def test_append_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    record = {"scheduler": "rl", "round": 1}

    append_jsonl(str(path), record)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == record
