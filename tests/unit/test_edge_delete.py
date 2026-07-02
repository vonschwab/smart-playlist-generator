# tests/unit/test_edge_delete.py
from src.playlist.repair.edge_delete import delete_broken_edges


def _score(pairs, default=0.9):
    """edge_score backed by a symmetric dict; default for unspecified pairs."""
    def score(a, b):
        return pairs.get((a, b), pairs.get((b, a), default))
    return score


def test_deletes_worse_interior_endpoint_and_merges():
    # piers 10,13 protected; edge 11->12 broken (0.05). Deleting 11 merges 10->12=0.7;
    # deleting 12 merges 11->13=0.6. Best = delete 11.
    score = _score({(11, 12): 0.05, (10, 12): 0.70, (11, 13): 0.60, (10, 11): 0.8, (12, 13): 0.8})
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 12, 13]
    assert len(r.delete_log) == 1 and r.delete_log[0]["deleted_idx"] == 11


def test_leaves_edge_when_no_deletion_improves():
    # broken 11->12=0.05; both merges also below the broken value -> never-worse blocks it.
    score = _score({(11, 12): 0.05, (10, 12): 0.02, (11, 13): 0.01})
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 11, 12, 13]
    assert r.delete_log == []


def test_never_deletes_between_two_piers():
    # broken edge 10->13 is directly between two protected piers -> cannot delete either.
    score = _score({(10, 13): 0.05})
    r = delete_broken_edges([10, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 13]
    assert r.delete_log == []


def test_noop_when_nothing_broken():
    score = _score({}, default=0.8)  # all edges 0.8 >= floor
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 11, 12, 13]
    assert r.delete_log == []


def test_respects_max_deletions():
    # two broken interior edges; cap at 1 deletion.
    score = _score({(11, 12): 0.05, (12, 13): 0.05, (10, 12): 0.7, (11, 13): 0.7,
                    (10, 11): 0.8, (13, 14): 0.8}, default=0.8)
    r = delete_broken_edges([10, 11, 12, 13, 14], edge_score=score, floor=0.30,
                            protected_indices={10, 14}, max_deletions=1)
    assert len(r.delete_log) == 1
