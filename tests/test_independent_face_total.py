import pytest

from app.independent_face_total import WEIGHTS, aggregate, total


def dimensions(values=None):
    values = values or [8.8, 8.5, 8.7, 8.4, 8.1]
    return [{"key": k, "score": v, "status": "assessable"} for k, v in zip(WEIGHTS, values)]


def test_weighted_total_uses_only_independent_scores_and_decimal_rounding():
    assert total(dimensions()) == 85.5  # 85.45, half up
    assert total(dimensions([0] * 5)) == 0
    assert total(dimensions([10] * 5)) == 100


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), -1, 11])
def test_missing_invalid_values_do_not_get_partial_or_default_total(bad):
    dims = dimensions(); dims[0]["score"] = bad
    assert total(dims) is None


def test_duplicate_missing_and_insufficient_items_are_rejected():
    with pytest.raises(ValueError): total(dimensions()[:-1])
    dims = dimensions(); dims[0]["key"] = "skin"
    with pytest.raises(ValueError): total(dims)
    dims = dimensions(); dims[0]["status"] = "insufficient"
    assert total(dims) is None


def test_ranks_are_continuous_stable_and_not_taken_from_old_overall():
    samples = [{"id": k, "label": k, "created_at": t} for k, t in [("b", "1"), ("a", "1"), ("c", "0")]]
    outcomes = {s["id"]: {"dimensions": dimensions(), "total": 99 if s["id"] == "b" else 1} for s in samples}
    rows = aggregate(samples, outcomes)
    assert [r["id"] for r in rows] == ["c", "a", "b"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
