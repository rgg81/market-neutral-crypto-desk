"""Phase 6 / Task 6.2 — new neutral lesson dimensions + DSR-gated promotion.

The corpus is re-keyed on market-neutral ALPHA (return net of BTC-beta), so the Reflector now
mints lessons tagged with a `dimension` drawn from the neutral failure modes:
`cointegration_break`, `carry_thesis_miss`, `neutrality_breach`, `sentiment_detract`. The genuine
code change under test is a TAG-AWARE retrieval filter: `score_lesson`/`retrieve_lessons` must read
the new `dimension` tag so a query for a dimension surfaces the matching lesson above an untagged
one (a behavioral filter, not a round-trip echo). The DSR gate on `statistically_promote` is
retained unchanged (promotion only when `dsr_pvalue >= 0.95`)."""

from datetime import UTC, datetime, timedelta

from futures_fund.contracts import Lesson
from futures_fund.lessons import (
    append_lesson,
    read_lessons,
    retrieve_lessons,
    score_lesson,
    statistically_promote,
)

NEW_DIMENSIONS = [
    "cointegration_break",
    "carry_thesis_miss",
    "neutrality_breach",
    "sentiment_detract",
]


def _L(text, *, dimension=None, polarity="restrictive", regime=None, tags=("risk",),
       importance=5, state="candidate"):
    f = {"text": text, "polarity": polarity, "regime": regime, "tags": list(tags),
         "importance": importance, "state": state}
    if dimension is not None:
        f["dimension"] = dimension
    return f


def test_dimension_field_roundtrips(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    append_lesson(tmp_path, _L("pair decohered", dimension="cointegration_break"), ts=now)
    assert read_lessons(tmp_path)[0].dimension == "cointegration_break"


def test_score_lesson_reads_dimension_tag():
    """A dimension query matches a lesson via its `dimension` field even when its `tags` do not
    overlap the query — the new tag-aware behavior. Without the code change the two lessons score
    identically (no `dimension` term), so this assertion is what fails first."""
    now = datetime(2026, 6, 2, tzinfo=UTC)
    tagged = Lesson(id="a", ts=now - timedelta(hours=1), text="pair broke",
                    tags=["misc"], dimension="cointegration_break")
    untagged = Lesson(id="b", ts=now - timedelta(hours=1), text="generic",
                      tags=["misc"], dimension=None)
    s_tagged = score_lesson(tagged, now, ["cointegration_break"])
    s_untagged = score_lesson(untagged, now, ["cointegration_break"])
    assert s_tagged > s_untagged


def test_retrieve_ranks_dimension_match_above_untagged(tmp_path):
    """`retrieve_lessons(query_tags=["cointegration_break"])` ranks the dimension-tagged lesson
    above an otherwise-identical untagged one (behavioral filter, not a round-trip)."""
    now = datetime(2026, 6, 2, tzinfo=UTC)
    append_lesson(tmp_path, _L("dimension lesson", dimension="cointegration_break",
                               tags=["misc"]), ts=now - timedelta(hours=1))
    append_lesson(tmp_path, _L("untagged lesson", tags=["misc"]), ts=now - timedelta(hours=1))
    got = retrieve_lessons(tmp_path, now=now, regime="x",
                           query_tags=["cointegration_break"], k=5)
    texts = [lz.text for lz in got]
    assert texts.index("dimension lesson") < texts.index("untagged lesson")


def test_all_new_dimensions_are_honored_under_polarity_quota(tmp_path):
    """Each of the four neutral dimensions is a recognized query value, and the polarity quota
    stays two-sided (an enabling lesson is force-included; restrictive fills are capped)."""
    now = datetime(2026, 6, 2, tzinfo=UTC)
    for dim in NEW_DIMENSIONS:
        append_lesson(tmp_path, _L(f"restrict-{dim}", dimension=dim, polarity="restrictive",
                                   importance=9, tags=["misc"]), ts=now - timedelta(hours=1))
    append_lesson(tmp_path, _L("DO re-enter the carry once funding flips", polarity="enabling",
                               dimension="carry_thesis_miss", importance=4, tags=["misc"]),
                  ts=now - timedelta(hours=40))
    for dim in NEW_DIMENSIONS:
        got = retrieve_lessons(tmp_path, now=now, regime="x", query_tags=[dim], k=5)
        # the matching dimension lesson surfaces to the top
        assert got[0].dimension == dim or got[0].state == "validated"
        assert any(lz.dimension == dim for lz in got), (dim, [lz.text for lz in got])
        # quota stays two-sided
        assert any(lz.polarity == "enabling" for lz in got)
        assert sum(1 for lz in got if lz.polarity == "restrictive") <= 3


def test_statistically_promote_requires_dsr_threshold(tmp_path):
    """DSR gate retained: a candidate at the count threshold is promoted only when
    `dsr_pvalue >= 0.95`; below the gate the confirmation still counts but it stays candidate."""
    now = datetime(2026, 6, 1, tzinfo=UTC)
    lid = append_lesson(tmp_path, _L("carry thesis missed", dimension="carry_thesis_miss"), ts=now)
    # four confirmations below the count threshold
    for _ in range(4):
        statistically_promote(tmp_path, lid, dsr_pvalue=0.99)
    assert next(lz for lz in read_lessons(tmp_path) if lz.id == lid).state == "candidate"
    # fifth confirmation but DSR below the gate -> still candidate
    assert statistically_promote(tmp_path, lid, dsr_pvalue=0.80) is True
    lz = next(lz for lz in read_lessons(tmp_path) if lz.id == lid)
    assert lz.state == "candidate" and lz.confirmations == 5
    # another confirmation WITH DSR support -> promoted
    statistically_promote(tmp_path, lid, dsr_pvalue=0.96)
    assert next(lz for lz in read_lessons(tmp_path) if lz.id == lid).state == "validated"
