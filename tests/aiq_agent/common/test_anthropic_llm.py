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


def test_minimax_active_config_uses_no_thinking_for_latency_sensitive_roles():
    config_path = Path("configs/config_cli_minimax_ddgs.yml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    llms = config["llms"]
    functions = config["functions"]

    assert llms["minimax_m3_fast_llm"]["thinking"] == {"type": "disabled"}
    assert llms["minimax_m3_research_llm"]["thinking"] == {"type": "disabled"}
    assert functions["intent_classifier"]["llm"] == "minimax_m3_fast_llm"
    assert functions["clarifier_agent"]["llm"] == "minimax_m3_fast_llm"
    assert functions["clarifier_agent"]["planner_llm"] == "minimax_m3_fast_llm"
    assert functions["shallow_research_agent"]["llm"] == "minimax_m3_research_llm"
    assert functions["deep_research_agent"]["researcher_llm"] == "minimax_m3_research_llm"
    assert functions["deep_research_agent"]["orchestrator_llm"] == "minimax_m3_synthesis_llm"
    assert functions["deep_research_agent"]["planner_llm"] == "minimax_m3_planner_llm"
