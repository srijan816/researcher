from pydantic import BaseModel

from aiq_agent.common.llm_json import parse_with_repair


class _DemoSchema(BaseModel):
    name: str
    count: int


async def test_parse_with_repair_accepts_direct_json():
    parsed = await parse_with_repair('{"name": "alpha", "count": 3}', _DemoSchema)

    assert parsed == _DemoSchema(name="alpha", count=3)


async def test_parse_with_repair_strips_markdown_fence():
    parsed = await parse_with_repair('```json\n{"name": "alpha", "count": 3}\n```', _DemoSchema)

    assert parsed == _DemoSchema(name="alpha", count=3)


async def test_parse_with_repair_uses_repair_callable():
    async def repair(_prompt: str) -> str:
        return '{"name": "fixed", "count": 4}'

    parsed = await parse_with_repair("not json", _DemoSchema, repair_llm=repair)

    assert parsed == _DemoSchema(name="fixed", count=4)


async def test_parse_with_repair_returns_none_when_unrepairable():
    parsed = await parse_with_repair("not json", _DemoSchema, repair_llm=lambda _prompt: "still not json")

    assert parsed is None
