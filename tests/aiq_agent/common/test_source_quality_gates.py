from aiq_agent.common.source_quality_gates import evaluate_source_quality


def test_source_quality_gates_pass_clean_distribution():
    report = evaluate_source_quality(
        [
            "https://bls.gov/report",
            "https://gemconsortium.org/report",
            "https://weforum.org/report",
            "https://idc.com/report",
            "https://openrouter.ai/models/example",
        ],
        tier="deep",
    )

    assert report.overall_status == "pass"
    assert report.distinct_domains == 5
    assert report.gate_results["authority_floor"] == "pass"


def test_source_quality_gates_fail_content_marketing_dominance():
    report = evaluate_source_quality(
        [
            "https://themoneypocket.com/a",
            "https://themoneypocket.com/b",
            "https://ideaproof.io/a",
            "https://packapop.com/a",
            "https://onlinekormo.com/a",
        ],
        tier="deep",
    )

    assert report.overall_status == "fail"
    assert report.gate_results["authority_floor"] == "fail"
    assert report.gate_results["weak_source_concentration"] == "fail"


def test_source_quality_gates_warn_on_single_domain_concentration():
    report = evaluate_source_quality(
        [
            "https://bls.gov/a",
            "https://bls.gov/b",
            "https://bls.gov/c",
            "https://bls.gov/d",
            "https://gemconsortium.org/report",
            "https://weforum.org/report",
            "https://idc.com/report",
        ],
        tier="deep",
    )

    assert report.overall_status == "warn"
    assert report.gate_results["concentration"] == "warn"


def test_source_quality_gates_use_shallow_thresholds():
    report = evaluate_source_quality(
        [
            "https://themoneypocket.com/a",
            "https://shopify.com/blog/b",
        ],
        tier="shallow",
    )

    assert report.gate_results["domain_diversity"] == "pass"
    assert report.gate_results["weak_source_concentration"] == "warn"
