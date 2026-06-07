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

    assert config.source_target == "40-77"
    assert config.advanced_web_search_limit == 92
    assert config.nominal_advanced_web_search_limit == 77
    assert config.search_calls_per_task == 16
    assert config.web_search_limit == 29

    profile = config.budget_profile(section_count=4)
    assert profile["total_search_calls"] == 92
    assert profile["nominal_search_calls"] == 77
    assert profile["adaptive_reserve_search_calls"] == 15
    assert profile["active_search_calls"] == 65
    assert profile["reserve_search_calls"] == 27


def test_deeper_tier_can_try_four_parallel_researchers():
    assert get_research_depth_config("deeper").max_parallel_researcher_tasks == 4
    assert get_research_depth_config("deep").max_parallel_researcher_tasks == 3


def test_non_shallow_extra_budget_preserves_normal_active_research_speed():
    medium_profile = get_research_depth_config("medium").budget_profile(section_count=4)
    deep_profile = get_research_depth_config("deep").budget_profile(section_count=4)

    assert medium_profile["total_search_calls"] == 62
    assert medium_profile["active_search_calls"] == 43
    assert medium_profile["reserve_search_calls"] == 19
    assert deep_profile["total_search_calls"] == 168
    assert deep_profile["active_search_calls"] == 119
    assert deep_profile["reserve_search_calls"] == 49


def test_standard_alias_routes_to_medium_latency_first_tier():
    assert normalize_research_depth("standard") == "medium"
