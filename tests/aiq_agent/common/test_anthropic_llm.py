from pathlib import Path

import yaml

from aiq_agent.common.anthropic_llm import AnthropicCompatibleModelConfig


def test_anthropic_compatible_config_preserves_thinking_controls():
    config = AnthropicCompatibleModelConfig(
        model_name="MiniMax-M3",
        base_url="https://api.minimax.io/anthropic",
        thinking={"type": "disabled"},
    )

    assert config.thinking == {"type": "disabled"}
    assert config.model_dump(exclude_none=True)["thinking"] == {"type": "disabled"}


def test_minimax_active_config_uses_m3_with_selective_orchestrator_thinking():
    config_path = Path("configs/config_cli_minimax_ddgs.yml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    llms = config["llms"]
    functions = config["functions"]
    config_text = config_path.read_text(encoding="utf-8")

    for legacy_model_marker in ("MiniMax-" + "M2", "m" + "27"):
        assert legacy_model_marker not in config_text
    assert llms["minimax_m3_fast_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_research_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_planner_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_planner_fast_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_synthesis_fast_llm"]["thinking"] == {"type": "disabled"}
    assert "MiniMax-M3" in llms["minimax_m3_deeper_planner_fast_llm"]["model_name"]
    assert "MiniMax-M3" in llms["minimax_m3_deeper_research_fast_llm"]["model_name"]
    assert llms["minimax_m3_deeper_planner_fast_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_deeper_research_fast_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_synthesis_llm"]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert not any(key.startswith("minimax_m3_deep_") for key in llms)
    assert functions["intent_classifier"]["llm"] == "minimax_m3_fast_llm"
    assert functions["clarifier_agent"]["llm"] == "minimax_m3_fast_llm"
    assert functions["clarifier_agent"]["planner_llm"] == "minimax_m3_fast_llm"
    assert functions["shallow_research_agent"]["llm"] == "minimax_m3_research_llm"
    assert functions["deep_research_agent"]["researcher_llm"] == "minimax_m3_research_llm"
    assert functions["deep_research_agent"]["orchestrator_llm"] == "minimax_m3_synthesis_llm"
    assert functions["deep_research_agent"]["medium_orchestrator_llm"] == "minimax_m3_synthesis_fast_llm"
    assert functions["deep_research_agent"]["deeper_orchestrator_llm"] == "minimax_m3_synthesis_fast_llm"
    assert functions["deep_research_agent"]["deeper_planner_llm"] == "minimax_m3_deeper_planner_fast_llm"
    assert functions["deep_research_agent"]["deeper_researcher_llm"] == "minimax_m3_deeper_research_fast_llm"
    assert "deep_planner_llm" not in functions["deep_research_agent"]
    assert "deep_researcher_llm" not in functions["deep_research_agent"]
    assert functions["deep_research_agent"]["planner_llm"] == "minimax_m3_planner_fast_llm"
