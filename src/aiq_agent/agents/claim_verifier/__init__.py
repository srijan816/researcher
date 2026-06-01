"""Claim-to-report verification helpers for deep research."""

from .agent import AlignmentResult
from .agent import CoverageResult
from .agent import ReportAtomicClaim
from .agent import VerificationReport
from .agent import VerificationSummary
from .agent import verify_report_claims_against_table

__all__ = [
    "AlignmentResult",
    "CoverageResult",
    "ReportAtomicClaim",
    "VerificationReport",
    "VerificationSummary",
    "verify_report_claims_against_table",
]
