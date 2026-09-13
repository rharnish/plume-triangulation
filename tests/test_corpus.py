"""Corpus selection and contamination tiers.

The property that matters most is the default: with FIGLIB_CORPUS unset, every stage must
read and write exactly the paths it did before corpora existed, so separately fetched
archives cannot change a published number by being on disk.
"""

from __future__ import annotations

import pytest

from src.figlib import corpus as C


def test_default_corpus_is_core_at_the_original_paths(monkeypatch):
    monkeypatch.delenv("FIGLIB_CORPUS", raising=False)
    c = C.current()
    assert c.name == "core"
    assert c.tgz_dirs == (C.ROOT / "data" / "tgz",)
    assert c.meta == C.ROOT / "data" / "meta"
    assert c.dets == C.ROOT / "out" / "yolo"
    assert c.out == C.ROOT / "out"


def test_extra_and_all_never_write_core_locations(monkeypatch):
    core = C.CORPORA["core"]
    for name in ("extra", "all"):
        monkeypatch.setenv("FIGLIB_CORPUS", name)
        c = C.current()
        assert c.meta != core.meta and c.dets != core.dets and c.out != core.out


def test_unknown_corpus_is_an_error(monkeypatch):
    monkeypatch.setenv("FIGLIB_CORPUS", "everything")
    with pytest.raises(ValueError):
        C.current()


@pytest.mark.parametrize("name, tier", [
    ("20240701_Kitchenfire_pi-e-mobo-c", "possibly_seen"),
    ("20250414_SnapshotDay", "possibly_seen"),       # the snapshot's own date
    ("20250415_DayAfter", "likely_unseen"),
    ("20260525_ReleaseDay", "likely_unseen"),        # the model's own upload date
    ("20260526_DayAfter", "unseen"),
    ("20260629_JunctionFire.2", "unseen"),
])
def test_contamination_tiers_by_fire_date(name, tier):
    assert C.contamination(name) == tier


def test_tier_filter(monkeypatch):
    monkeypatch.delenv("FIGLIB_TIER", raising=False)
    assert C.tier_filter() is None and C.tier_suffix(None) == ""
    monkeypatch.setenv("FIGLIB_TIER", "likely_unseen,unseen")
    tiers = C.tier_filter()
    assert tiers == ("likely_unseen", "unseen")
    assert C.in_tier("20250709_SteeleFire", tiers)
    assert not C.in_tier("20240701_Kitchenfire", tiers)
    assert C.tier_suffix(tiers) == "_likely_unseen+unseen"
    monkeypatch.setenv("FIGLIB_TIER", "clean")
    with pytest.raises(ValueError):
        C.tier_filter()


def test_hyphen_joined_sequence_names_split_without_changing_underscore_ones(cams):
    from src.figlib.ingest import split_seq_name
    assert split_seq_name("20190814_FIRE-pi-s-mobo-c", cams) == ("20190814_FIRE", "pi-s-mobo-c")
    assert split_seq_name("20240701_Kitchenfire_pi-e-mobo-c", cams) == (
        "20240701_Kitchenfire", "pi-e-mobo-c")
    # An event name that itself contains hyphens still splits on the underscore.
    assert split_seq_name("20201202_WillowFire-nightime-near-CDF-HQ_sm-n-mobo-c", cams)[0] \
        == "20201202_WillowFire-nightime-near-CDF-HQ"


def test_frame_names_with_an_encoded_sign_resolve_against_t0():
    from src.figlib.ingest import resolve_frame_names
    t0 = 1754084174
    d = "Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/seq/"
    pos = [f"{d}{t0 + o}_+{o:05d}.jpg" for o in (0, 60)]
    neg = [f"{d}{t0 - o}_-{o:05d}.jpg" for o in (60, 120)]
    enc = [f"{d}{t0 - o}_%%2B{o:05d}.jpg" for o in (60, 120)]

    # Bernardo: encoded copies of frames already present with a plain '-'.
    frames, stats = resolve_frame_names(pos + neg + enc + [f"{d}index.html"])
    assert sorted(frames.values()) == sorted([(t0 - 120, -120), (t0 - 60, -60),
                                              (t0, 0), (t0 + 60, 60)])
    assert stats == {"sign_repaired": 0, "duplicates": 2, "unresolved": 0}

    # Cool: the encoded frames are the only negatives, and are recovered.
    frames, stats = resolve_frame_names(pos + enc)
    assert sorted(o for _, o in frames.values()) == [-120, -60, 0, 60]
    assert stats["sign_repaired"] == 2

    # Nothing plain to anchor t0: an encoded sign cannot be trusted either way.
    frames, stats = resolve_frame_names(enc)
    assert frames == {} and stats["unresolved"] == 2


def test_one_epoch_under_two_annotation_passes_is_not_a_duplicate():
    from src.figlib.ingest import resolve_frame_names
    names = ["s/1700000000_+00000.jpg", "s/1700000000_-00360.jpg"]
    frames, stats = resolve_frame_names(names)
    assert len(frames) == 2 and stats["duplicates"] == 0
