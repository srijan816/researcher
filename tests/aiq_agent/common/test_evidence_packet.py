import json

from aiq_agent.common.evidence_packet import build_evidence_packet


def test_build_evidence_packet_ranks_authoritative_claim_evidence_first():
    claim_table = {
        "atomic_claims": [
            {
                "claim_id": "C1",
                "claim_text": "Official docs state the model has a large context window.",
                "claim_type": "specification",
                "expected_answer_shape": "number",
                "preferred_source_classes": ["first_party"],
                "status": "verified",
                "resolved_value": "1M tokens",
                "evidence": [
                    {
                        "source_url": "https://docs.minimax.io/model/m3",
                        "source_class": "first_party",
                        "extract": "MiniMax M3 supports a one million token context window.",
                        "extract_confidence": "high",
                    }
                ],
            },
            {
                "claim_id": "C2",
                "claim_text": "A blog says M3 is useful for research.",
                "claim_type": "recommendation",
                "expected_answer_shape": "free_text",
                "preferred_source_classes": ["any_credible"],
                "status": "partially_verified",
                "resolved_value": "Useful for research",
                "evidence": [
                    {
                        "source_url": "https://example-blog.com/m3-research",
                        "source_class": "content_marketing",
                        "extract": "The blog says M3 can help research workflows.",
                        "extract_confidence": "low",
                    }
                ],
                "hedge_required": True,
                "hedge_phrase": "according to secondary commentary",
            },
        ]
    }

    packet = build_evidence_packet(job_id="job-1", claim_table_content=json.dumps(claim_table))

    assert packet.job_id == "job-1"
    assert packet.claim_count == 2
    assert packet.source_count == 2
    assert packet.sources[0].source_class == "first_party"
    assert packet.sources[0].claim_ids == ["C1"]
    assert "one million token" in packet.sources[0].extracts[0].text


def test_build_evidence_packet_merges_extract_fragments_with_claim_sources():
    claim_table = {
        "atomic_claims": [
            {
                "claim_id": "C1",
                "claim_text": "Primary source supports the claim.",
                "claim_type": "quantitative",
                "expected_answer_shape": "number",
                "preferred_source_classes": ["primary_issuer"],
                "status": "verified",
                "resolved_value": "42%",
                "evidence": [
                    {
                        "source_url": "https://bls.gov/report",
                        "source_class": "primary_issuer",
                        "extract": "The source reports 42 percent.",
                        "extract_confidence": "high",
                    }
                ],
            }
        ]
    }
    extract_fragment = {
        "extracts": [
            {
                "claim_ids": ["C1"],
                "url": "https://bls.gov/report",
                "title": "BLS Report",
                "source_class": "primary_issuer",
                "extraction_status": "extracted",
                "extract": "A second exact extract from the same source.",
            }
        ]
    }

    packet = build_evidence_packet(
        claim_table_content=json.dumps(claim_table),
        extract_contents=[json.dumps(extract_fragment)],
    )

    assert packet.source_count == 1
    assert packet.sources[0].title == "BLS Report"
    assert len(packet.sources[0].extracts) == 2


def test_build_evidence_packet_drops_off_topic_registry_only_sources():
    class RegistrySource:
        def __init__(self, url: str, title: str, source_class: str = "academic"):
            self.url = url
            self.title = title
            self.source_class = source_class

    packet = build_evidence_packet(
        request_text="Research social movements, civil rights activism, protest tactics, and political change.",
        registry_sources=[
            RegistrySource("https://example.edu/qaidam-basin-hydroclimate", "Qaidam Basin hydroclimate"),
            RegistrySource(
                "https://journals.example.edu/civil-rights-protest-movements",
                "Civil rights protest movements and political change",
            ),
        ],
    )

    urls = [source.url for source in packet.sources]
    assert "https://journals.example.edu/civil-rights-protest-movements" in urls
    assert "https://example.edu/qaidam-basin-hydroclimate" not in urls
