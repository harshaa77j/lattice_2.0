import pytest
from shared.data import slice_data

N = 50000  

def all_slices(num_samples, num_slices, seed=42):
    return [slice_data(num_samples, i, num_slices, seed) for i in range(num_slices)]

@pytest.mark.parametrize("num_slices", [1, 2, 3, 5, 7])
def test_no_overlap(num_slices):
    seen = set()
    for s in all_slices(N, num_slices):
        assert seen.isdisjoint(s), "an index appears in two slices"
        seen.update(s)

@pytest.mark.parametrize("num_slices", [1, 2, 3, 5, 7])
def test_full_coverage(num_slices):
    covered = set().union(*all_slices(N, num_slices))
    assert covered == set(range(N)), "some indices are missing"

@pytest.mark.parametrize("num_slices", [2, 3, 5, 7])
def test_sizes_differ_by_at_most_one(num_slices):
    sizes = [len(s) for s in all_slices(N, num_slices)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == N

def test_no_duplicates_within_a_slice():
    s = slice_data(N, 0, 3)
    assert len(s) == len(set(s))

def test_same_seed_same_result():
    assert slice_data(N, 2, 5, seed=1) == slice_data(N, 2, 5, seed=1)

def test_different_seed_different_result():
    assert slice_data(N, 2, 5, seed=1) != slice_data(N, 2, 5, seed=2)

def test_one_slice_is_everything():
    assert sorted(slice_data(N, 0, 1)) == list(range(N))

def test_slices_are_shuffled():
    assert slice_data(N, 0, 5)[:5] != [0, 5, 10, 15, 20]

@pytest.mark.parametrize("slice_id, num_slices", [
    (5, 5),    # one past the last slice
    (-1, 5),   # negative
    (0, 0),    # zero slices
    (0, -3),   # negative number of slices
])
def test_invalid_slice_arguments(slice_id, num_slices):
    with pytest.raises(ValueError):
        slice_data(N, slice_id, num_slices)

def test_more_slices_than_samples():
    with pytest.raises(ValueError):
        slice_data(3, 0, 5)