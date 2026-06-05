from aiq_agent.common.source_scoring import score_source


def test_score_source_prefers_primary_sources():
    score = score_source(
        url="https://bls.gov/news.release/example-2026.htm",
        title="BLS 2026 release",
        source_class="primary_issuer",
        used_for=["C1", "C2"],
        extract_count=2,
    )

    assert score.authority == 5
    assert score.recency >= 4
    assert score.relevance == 4
    assert score.bias_risk == 1


def test_score_source_marks_unknown_as_high_bias_risk():
    score = score_source(url="https://random-example-site.test/article")

    assert score.source_class == "unknown"
    assert score.authority == 1
    assert score.bias_risk == 5
