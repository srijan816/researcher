from aiq_agent.common.claude_code_specialist import _build_prompt
from aiq_agent.common.claude_code_specialist import _claude_code_env
from aiq_agent.common.claude_code_specialist import should_use_claude_code


def test_should_use_claude_code_for_plan_director_boundary():
    decision = should_use_claude_code(
        "Create a synthesis-ready comparison matrix and critique missing sources",
        current_context_size=40_000,
        num_sources=12,
        stage="plan_director",
        tier="medium",
    )

    assert decision.should_use
    assert decision.score >= 2
    assert decision.reasons


def test_should_use_claude_code_for_scope_classifier_boundary():
    decision = should_use_claude_code(
        "Classify this broad slide-deck research request before planning",
        current_context_size=2000,
        num_sources=0,
        stage="scope_classifier",
        tier="deeper",
    )

    assert decision.should_use
    assert decision.score >= 2
    assert decision.reasons


def test_should_not_use_claude_code_for_simple_search():
    decision = should_use_claude_code(
        "Search one URL and summarize it",
        current_context_size=1000,
        num_sources=1,
        stage="research",
        tier="medium",
    )

    assert not decision.should_use


def test_claude_code_env_defaults_to_minimax(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-anthropic-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    env = _claude_code_env()

    assert env["ANTHROPIC_API_KEY"] == "mini-key"  # pragma: allowlist secret
    assert env["ANTHROPIC_BASE_URL"] == "https://api.minimax.io/anthropic"


def test_claude_code_prompt_requires_exact_plan_artifact_path(tmp_path):
    output_path = tmp_path / "plan_director.md"

    prompt = _build_prompt(
        task="Improve the plan direction without doing web research.",
        context="Original request and /shared/plan.json contents",
        stage="plan_director",
        output_path=output_path,
    )

    assert f"Required artifact path: {output_path.resolve()}" in prompt
    assert "The AI-Q orchestrator will ignore ordinary stdout" in prompt
    assert "# Claude Plan Direction Memo" in prompt
    assert "## Research Task Split" in prompt
    assert "Do not perform web research" in prompt


def test_claude_code_prompt_requires_exact_scope_artifact_path(tmp_path):
    output_path = tmp_path / "scope_classifier.md"

    prompt = _build_prompt(
        task="Classify the user request before planning.",
        context="Original request: research social movements to create WSDC/BP training.",
        stage="scope_classifier",
        output_path=output_path,
    )

    assert f"Required artifact path: {output_path.resolve()}" in prompt
    assert "# Claude Scope Classification Memo" in prompt
    assert "## Primary Subject" in prompt
    assert "## Forbidden Reframes" in prompt
    assert "## Research Modules" in prompt
