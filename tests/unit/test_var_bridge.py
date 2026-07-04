from src.playlist.pier_bridge.var_bridge import segment_bottleneck, choose_segment_length


def test_bottleneck_is_min_edge_and_its_index():
    # nodes 0..3; edge score = fixed table
    edges = {(0, 1): 0.8, (1, 2): 0.2, (2, 3): 0.9}
    b, idx = segment_bottleneck([0, 1, 2, 3], lambda a, c: edges[(a, c)])
    assert b == 0.2 and idx == 1            # weakest edge is 1->2 at position 1


def test_choose_keeps_nominal_when_already_good():
    calls = []
    def build(l):
        calls.append(l)
        return ([10] * l, 0.7)               # every length scores 0.7
    chosen_l, path, flexed = choose_segment_length(6, 4, 8, build, good_enough=0.5, eps=0.02)
    assert chosen_l == 6 and path == [10] * 6
    assert calls == [6]                       # nominal good enough -> no flex, ONE build
    assert flexed is False                    # nominal was good enough, did not flex


def test_choose_flexes_to_best_bottleneck_when_nominal_weak():
    scores = {4: 0.10, 5: 0.45, 6: 0.12, 7: 0.40, 8: 0.20}   # nominal 6 is weak
    def build(l):
        return ([l], scores[l])
    chosen_l, path, flexed = choose_segment_length(6, 4, 8, build, good_enough=0.5, eps=0.02)
    assert chosen_l == 5 and path == [5]      # 5 has the best bottleneck (0.45)
    assert flexed is True                     # nominal was weak, other lengths were evaluated


def test_choose_prefers_nominal_within_epsilon():
    scores = {5: 0.46, 6: 0.45, 7: 0.30}      # 5 best but 6 within eps -> keep 6
    def build(l):
        return ([l], scores[l])
    chosen_l, _, flexed = choose_segment_length(6, 5, 7, build, good_enough=0.9, eps=0.02)
    assert chosen_l == 6                       # nominal preferred when within eps of best
    assert flexed is True                      # nominal was below good_enough (0.9), flex was attempted


def test_choose_respects_band_clamp():
    # lo/hi narrower than nominal +/- flex: only 6,7 allowed
    scores = {6: 0.1, 7: 0.9}
    chosen_l, _, flexed = choose_segment_length(6, 6, 7, lambda l: ([l], scores[l]), good_enough=0.5, eps=0.02)
    assert chosen_l == 7
    assert flexed is True                      # nominal score 0.1 < good_enough 0.5, so flexed


def test_choose_no_flex_when_no_room_even_if_nominal_weak():
    # lo == hi == nominal: the flex cap has forced a single buildable length.
    # Nominal is weak (0.1 < good_enough) but there is nothing else to evaluate,
    # so flexed must be False (regression guard for the (N+1/N) flex-counter
    # over-count where non-flexed segments logged flexed=True with chosen==nominal).
    calls = []
    def build(l):
        calls.append(l)
        return ([l], 0.1)
    chosen_l, path, flexed = choose_segment_length(6, 6, 6, build, good_enough=0.5, eps=0.02)
    assert chosen_l == 6 and path == [6]
    assert calls == [6]                        # only nominal built — no extra beam work
    assert flexed is False                     # no alternative length existed -> did NOT flex
