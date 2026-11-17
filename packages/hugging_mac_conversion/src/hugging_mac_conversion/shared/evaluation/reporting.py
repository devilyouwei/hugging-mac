"""Readable evidence from the same data used for machine-readable evaluation reports."""

from __future__ import annotations

from typing import Any


def _cell(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['model']} / {report['variant']} evaluation",
        "",
        f"- Status: **{report['status']}**",
        f"- Scope: `{report['scope']}`; this is not complete-model or SDK acceptance.",
        f"- Source commit: `{report['source_revision']}`",
        f"- Original files: `{report['source_path']}`",
        f"- Source selections: `{', '.join(report['source_selections'])}`",
        f"- Cases SHA-256: `{report['cases_sha256']}`",
        "",
        "## Original model",
        "",
        "| Case | Capability | Status | Decision | Expected | Correct | Inference ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("reference_cases", []):
        lines.append(
            "| "
            + " | ".join(
                _cell(row.get(k))
                for k in (
                    "id",
                    "capability",
                    "status",
                    "decision",
                    "expected",
                    "correct",
                    "inference_ms",
                )
            )
            + " |"
        )
        if row.get("reason"):
            lines.extend(["", f"Case `{row['id']}`: {_cell(row['reason'])}", ""])
    config = report["evaluation_config"]
    lines.extend(
        [
            "",
            "## Converted models",
            "",
            f"Probability tolerances: atol={config['atol']}, rtol={config['rtol']}. "
            f"Require decision agreement: {config['require_decision_agreement']}.",
            "Parity and expected-answer correctness are separate measures. Timings are single "
            "observations, not a performance benchmark.",
        ]
    )
    if not report.get("candidates"):
        lines.extend(["", "No converted-model comparison was completed."])
    for candidate in report.get("candidates", []):
        lines.extend(
            [
                "",
                f"### {candidate['runtime']}: {candidate['status']}",
                "",
                f"- Bundle: `{candidate['run']}`",
                f"- Build status: `{candidate['build_status']}`",
                "",
                "| Case | Status | Max probability error | Decision | Agreement | Correct | "
                "Inference ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in candidate["cases"]:
            lines.append(
                "| "
                + " | ".join(
                    _cell(row.get(k))
                    for k in (
                        "id",
                        "status",
                        "max_absolute_error",
                        "converted_decision",
                        "decision_agreement",
                        "converted_correct",
                        "converted_ms",
                    )
                )
                + " |"
            )
        for row in [candidate, *candidate["cases"]]:
            if row.get("reason"):
                lines.extend(["", f"{_cell(row.get('id', 'Backend'))}: {_cell(row['reason'])}"])
    lines.extend(["", "## Capability coverage", "", "| Capability | Status |", "| --- | --- |"])
    for name, status in report["capability_coverage"].items():
        lines.append(f"| {_cell(name)} | {_cell(status)} |")
    lines.extend(
        [
            "",
            "Untested capabilities remain required; untested does not mean unsupported.",
            "",
            "## Tool versions",
            "",
        ]
    )
    for name, version in report.get("versions", {}).items():
        lines.append(f"- {name}: `{version}`")
    for name, value in report.get("environment", {}).items():
        lines.append(f"- {name}: `{_cell(value)}`")
    if report.get("reason"):
        lines.extend(["", f"Evaluation error: {_cell(report['reason'])}"])
    return "\n".join(lines) + "\n"
