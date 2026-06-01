from aiq_agent.common.claim_table import AtomicClaim
from aiq_agent.common.claim_table import ClaimEvidence
from aiq_agent.common.claim_table_io import CLAIM_TABLE_PATH
from aiq_agent.common.claim_table_io import merge_claim_files
from aiq_agent.common.claim_table_io import write_researcher_claims


class _MemoryFilesystem:
    def __init__(self):
        self.files = {}


async def test_claim_table_io_writes_per_researcher_fragments_and_merges():
    filesystem = _MemoryFilesystem()
    await write_researcher_claims(
        "researcher one",
        [
            AtomicClaim(
                claim_id="C1",
                claim_text="A vendor pricing fact",
                claim_type="quantitative",
                expected_answer_shape="number",
                preferred_source_classes=["first_party"],
                status="verified",
                resolved_value="$1",
                evidence=[
                    ClaimEvidence(
                        source_url="https://vendor.example/docs/pricing",
                        source_class="first_party",
                        extract="The price is $1.",
                    )
                ],
            )
        ],
        filesystem,
    )

    table = await merge_claim_files(filesystem, job_id="job-1")

    assert CLAIM_TABLE_PATH in filesystem.files
    assert table.job_id == "job-1"
    assert table.verified_count() == 1
