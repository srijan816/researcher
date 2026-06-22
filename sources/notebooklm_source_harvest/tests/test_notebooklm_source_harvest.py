from notebooklm_source_harvest import HarvestSource
from notebooklm_source_harvest import NotebookLMCLI
from notebooklm_source_harvest import NotebookLMSourceHarvestConfig
from notebooklm_source_harvest import _dedupe_sources
from notebooklm_source_harvest import _extract_notebook_id
from notebooklm_source_harvest import _is_generated_report
from notebooklm_source_harvest import _iter_source_like
from notebooklm_source_harvest import _source_from_dict


def test_extract_notebook_id_from_nested_payload():
    payload = {"notebook": {"projectId": "abc123"}}

    assert _extract_notebook_id(payload) == "abc123"


def test_iter_source_like_accepts_sources_envelope():
    payload = {"sources": [{"id": "s1", "title": "One"}, {"id": "s2", "title": "Two"}]}

    assert [item["id"] for item in _iter_source_like(payload)] == ["s1", "s2"]


def test_source_from_dict_normalizes_common_keys():
    source = _source_from_dict(
        {
            "sourceId": "s1",
            "displayTitle": "Research Source",
            "sourceUrl": "https://example.com/report",
            "state": "COMPLETED",
            "sourceType": "web",
        }
    )

    assert source.source_id == "s1"
    assert source.title == "Research Source"
    assert source.url == "https://example.com/report"
    assert source.status == "ready"
    assert source.source_type == "web"


def test_generated_deep_research_report_is_detected():
    source = HarvestSource(
        source_id="r1",
        title="Generated Markdown deep-research report",
        url="",
        status="ready",
        source_type="notebooklm",
    )

    assert _is_generated_report(source)


def test_dedupe_prefers_ready_source_over_retry_error():
    ready = HarvestSource(
        source_id="s1",
        title="Medium article",
        url="https://medium.com/example/article",
        status="ready",
        source_type="web",
    )
    error = HarvestSource(
        source_id="s2",
        title="Medium article retry",
        url="https://medium.com/example/article",
        status="error",
        source_type="web",
        error="import failed",
    )

    assert _dedupe_sources([error, ready]) == [ready]


def test_list_sources_uses_caller_timeout(tmp_path):
    config = NotebookLMSourceHarvestConfig(work_dir=str(tmp_path))
    client = NotebookLMCLI(config)
    observed = {}

    def fake_run(args, *, timeout=None):
        import subprocess

        observed["args"] = args
        observed["timeout"] = timeout
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"sources":[{"id":"s1","title":"One","url":"https://example.com","status":"ready"}]}',
            stderr="",
        )

    client.run = fake_run

    sources = client.list_sources("notebook-1", timeout=7)

    assert observed["args"] == ["source", "list", "--notebook", "notebook-1", "--json"]
    assert observed["timeout"] == 7
    assert sources[0].status == "ready"
