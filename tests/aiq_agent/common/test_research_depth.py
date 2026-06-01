from aiq_agent.common.research_depth import get_research_depth_config
from aiq_agent.common.research_depth import normalize_research_depth


def test_medium_is_first_class_depth_tier():
    config = get_research_depth_config("medium")

    assert normalize_research_depth("medium") == "medium"
    assert config.tier == "medium"
    assert config.advanced_web_search_limit == 64
    assert config.max_parallel_researcher_tasks == 3


def test_m3_heavy_tiers_use_three_parallel_researchers():
    assert get_research_depth_config("deeper").max_parallel_researcher_tasks == 3
    assert get_research_depth_config("deep").max_parallel_researcher_tasks == 3


def test_standard_alias_routes_to_medium_latency_first_tier():
    assert normalize_research_depth("standard") == "medium"
