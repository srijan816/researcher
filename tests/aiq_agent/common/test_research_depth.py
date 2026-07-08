from aiq_agent.common.research_depth import get_research_depth_config
from aiq_agent.common.research_depth import normalize_research_depth


def test_medium_is_first_class_depth_tier():
    config = get_research_depth_config("medium")

    assert normalize_research_depth("medium") == "medium"
    assert config.tier == "medium"
    assert config.advanced_web_search_limit == 62
    assert config.nominal_advanced_web_search_limit == 51
    assert config.search_calls_per_task == 11
    assert config.max_researcher_tasks == 4
    assert config.max_parallel_researcher_tasks == 3


def test_deeper_tier_has_adaptive_reserve_budget():
    config = get_research_depth_config("deeper")

    assert config.source_target == "40-70"
    assert config.advanced_web_search_limit == 70
    assert config.nominal_advanced_web_search_limit == 60
    assert config.search_calls_per_task == 10
    assert config.web_search_limit == 20

    profile = config.budget_profile(section_count=4)
    assert profile["total_search_calls"] == 70
    assert profile["nominal_search_calls"] == 60
    assert profile["adaptive_reserve_search_calls"] == 10
    assert profile["active_search_calls"] == 51
    assert profile["reserve_search_calls"] == 19


def test_deeper_tier_can_try_three_parallel_researchers():
    assert get_research_depth_config("deeper").max_parallel_researcher_tasks == 3
    assert get_research_depth_config("deep").max_parallel_researcher_tasks == 3


def test_non_shallow_extra_budget_preserves_normal_active_research_speed():
    medium_profile = get_research_depth_config("medium").budget_profile(section_count=4)
    deep_profile = get_research_depth_config("deep").budget_profile(section_count=4)

    assert medium_profile["total_search_calls"] == 62
    assert medium_profile["active_search_calls"] == 43
    assert medium_profile["reserve_search_calls"] == 19
    assert deep_profile["total_search_calls"] == 106
    assert deep_profile["active_search_calls"] == 76
    assert deep_profile["reserve_search_calls"] == 30


def test_standard_alias_routes_to_medium_latency_first_tier():
    assert normalize_research_depth("standard") == "medium"
