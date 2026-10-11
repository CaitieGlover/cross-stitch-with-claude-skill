import numpy as np
import pytest

from crossstitch import dmc_match as d


def test_table_loaded():
    threads = d.dmc_threads()
    assert len(threads) > 400
    assert len({t.number for t in threads}) == len(threads)
    assert all(len(t.hex) == 6 for t in threads)


def test_lookup_accepts_common_spellings():
    assert d.lookup("310").name == "Black"
    assert d.lookup("DMC 310").hex == "000000"
    assert d.lookup("ecru").number == "Ecru"
    assert d.lookup("blanc").number == "White"
    with pytest.raises(KeyError):
        d.lookup("99999")


def test_exact_colours_match_themselves():
    for number in ("310", "321", "B5200", "3865", "798"):
        t = d.lookup(number)
        match, dist = d.nearest_one(t.rgb)
        assert match == t or dist == 0
        assert dist == pytest.approx(0, abs=1e-9)


def test_ciede2000_reference_values():
    # Pairs and expected values from Sharma, Wu & Dalal (2005).
    pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]
    for a, b, expected in pairs:
        assert float(d.delta_e2000(np.array(a), np.array(b))) == pytest.approx(expected, abs=1e-4)


def test_nearest_respects_candidate_list():
    stash = [d.lookup("310"), d.lookup("B5200")]
    match, _ = d.nearest_one((40, 40, 40), stash)
    assert match.number == "310"
    match, _ = d.nearest_one((230, 230, 230), stash)
    assert match.number == "B5200"


def test_rgb_metric_available():
    match, dist = d.nearest_one((0, 0, 0), metric="rgb")
    assert match.number == "310" and dist == 0
