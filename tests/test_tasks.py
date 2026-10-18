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


def test_store_loads_shipped_tasks() -> None:
    store = TaskStore(Path("env/tasks/toy_v1.jsonl"))
    assert len(store) == 200
    assert len(store.list(split="train")) == 160
    assert len(store.list(split="heldout")) == 40
    assert len(store.list(split="train", n=5, seed=1)) == 5
    assert store.list(split="train", n=5, seed=1) == store.list(split="train", n=5, seed=1)
    t = store.list(n=1)[0]
    assert store.ground_truth(t.task_id) is not None
    assert store.ground_truth("nope") is None
