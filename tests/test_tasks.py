"""Task generation is deterministic and the store hides ground truth."""

from pathlib import Path

from env.tasks.generate import FIELDS, generate
from env.tasks.store import TaskStore


def test_generate_deterministic_and_split() -> None:
    a = generate(seed=7, n_total=20, n_train=16, n_max=4)
    b = generate(seed=7, n_total=20, n_train=16, n_max=4)
    assert a == b
    assert sum(r["split"] == "train" for r in a) == 16
    assert generate(seed=8, n_total=20, n_train=16, n_max=4) != a
    for r in a:
        assert set(r["ground_truth"]) == set(FIELDS)
        assert "ground_truth" not in r["task"]["context"]
        assert set(r["task"]["context"]["lookup"]) >= {"sample_id", "temperature", "concentration", "method"}


def test_hard_tasks_have_decoys_and_arithmetic() -> None:
    recs = generate(seed=7, n_total=60, n_train=48, n_max=4, difficulty="hard")
    texts = " ".join(r["task"]["context"]["notebook_text"] for r in recs)
    assert "diluted 1:" in texts and "correction" in texts and "Pre-incubated" in texts
    assert "kDa" in texts and "Run as a series" in texts
    assert all("storage_temperature" in r["task"]["context"]["lookup"] for r in recs)
    easy = generate(seed=7, n_total=60, n_train=48, n_max=4, difficulty="easy")
    assert all("storage_temperature" not in r["task"]["context"]["lookup"] for r in easy)


def test_store_loads_shipped_tasks() -> None:
    store = TaskStore(Path("env/tasks/toy_v2.jsonl"))
    assert len(store) == 200
    assert len(store.list(split="train")) == 160
    assert len(store.list(split="heldout")) == 40
    assert len(store.list(split="train", n=5, seed=1)) == 5
    assert store.list(split="train", n=5, seed=1) == store.list(split="train", n=5, seed=1)
    t = store.list(n=1)[0]
    assert store.ground_truth(t.task_id) is not None
    assert store.ground_truth("nope") is None


def test_gsm8k_record_builder_without_download() -> None:
    from env.tasks.hf_gsm8k import parse_answer, to_record
    assert parse_answer("Natalia sold 48/2 = 24 clips.\n#### 72") == 72.0
    assert parse_answer("... #### 1,250.5") == 1250.5
    r = to_record({"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}, 0, "train", 6)
    assert r["ground_truth"] == {"answer": 4.0} and r["task"]["context"]["tool"]["name"] == "calculator"
    assert r["task"]["context"]["domain"] == "gsm8k"
