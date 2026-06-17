from intervals import merge


def test_basic_overlap():
    assert merge([[1, 3], [2, 6]]) == [[1, 6]]


def test_touching_merges():
    # Touching intervals (end == next start) must merge.
    assert merge([[1, 2], [2, 3]]) == [[1, 3]]


def test_nested_keeps_outer_end():
    # A fully-nested interval must not shrink the outer interval's end.
    assert merge([[1, 5], [2, 3]]) == [[1, 5]]


def test_disjoint_unchanged():
    assert merge([[1, 2], [5, 6]]) == [[1, 2], [5, 6]]


def test_unsorted_input():
    assert merge([[5, 6], [1, 4], [2, 3]]) == [[1, 4], [5, 6]]


def test_chain_merges():
    assert merge([[1, 4], [4, 7], [7, 9]]) == [[1, 9]]


def test_empty():
    assert merge([]) == []
