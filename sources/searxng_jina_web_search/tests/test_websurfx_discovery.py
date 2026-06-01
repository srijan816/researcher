from searxng_jina_web_search.register import SearXNGJinaWebSearchToolConfig
from searxng_jina_web_search.register import _normalize_websurfx_result


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
    )

    assert config.discovery_backend == "websurfx"
    assert config.websurfx_url == "http://websurfx:8080"
    assert config.websurfx_engines == "Brave,Wikipedia"
