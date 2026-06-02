from searxng_jina_web_search.register import SearXNGJinaWebSearchToolConfig
from searxng_jina_web_search.register import _normalize_websurfx_result
from searxng_jina_web_search.register import simplify_search_query


def test_normalize_websurfx_result_maps_camel_case_fields() -> None:
    result = _normalize_websurfx_result(
        {
            "title": "Example result",
            "url": "https://example.com/report",
            "description": "A concise result description.",
            "engine": ["DuckDuckGo", "Bing"],
            "relevanceScore": 0.87,
        }
    )

    assert result == {
        "title": "Example result",
        "url": "https://example.com/report",
        "content": "A concise result description.",
        "engines": ["websurfx:DuckDuckGo", "websurfx:Bing"],
        "score": 0.87,
        "_discovery_backend": "websurfx",
    }


def test_normalize_websurfx_result_rejects_missing_url() -> None:
    assert _normalize_websurfx_result({"title": "No URL"}) is None


def test_config_accepts_websurfx_discovery_backend() -> None:
    config = SearXNGJinaWebSearchToolConfig(
        discovery_backend="websurfx",
        websurfx_url="http://websurfx:8080",
        websurfx_engines="Brave,Wikipedia",
        websurfx_timeout_seconds=8,
    )

    assert config.discovery_backend == "websurfx"
    assert config.websurfx_url == "http://websurfx:8080"
    assert config.websurfx_engines == "Brave,Wikipedia"
    assert config.websurfx_timeout_seconds == 8


def test_simplify_search_query_reduces_task_packet() -> None:
    packet = """
    Research authoritative evidence for the assigned task.
    QUERY: social media algorithm liability regulatory capture comparative cases
    SEARCH STRATEGY:
    - target_claims: C1, C2, C3
    - search_budget: 12
    - acceptance criteria: cite sources and write files
    """

    query = simplify_search_query(packet)

    assert query == "social media algorithm liability regulatory capture comparative cases"
    assert "SEARCH STRATEGY" not in query
    assert "target_claims" not in query
    assert len(query) < 120


def test_simplify_search_query_prefers_seed_queries() -> None:
    packet = """
    query: Research all current evidence on whether broad regulatory-capture theory applies to
    social media algorithm liability and institutional design.
    seed_queries: regulatory capture theory social media algorithms 2026; platform liability institutional design
    target_claims: C1, C2
    search_budget: 12
    """

    query = simplify_search_query(packet)

    assert query == "regulatory capture theory social media algorithms 2026"


def test_simplify_search_query_preserves_short_query() -> None:
    assert simplify_search_query("AI automation ROI analyst reports 2026") == "AI automation ROI analyst reports 2026"
