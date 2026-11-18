"""Execute a complete Vela Core AI bundle and its original-reference test cases."""

import argparse
import json
from pathlib import Path
from typing import Any

from hugging_mac_conversion.models.vela.evaluation.comparison import compare
from hugging_mac_conversion.models.vela.utils.inference import VelaFullCoreAI


def validate_conversion(root: Path) -> dict[str, Any]:
    fixtures = json.loads((root / "validation_cases.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    checks = []
    with VelaFullCoreAI(root) as model:
        for case in fixtures:
            row: dict[str, Any] = {"id": case["id"], "passed": False}
            try:
                actual = model.system_one(case["state"], case["questions"])
                row["max_absolute_error"] = compare(actual, case["expected"])
                row["passed"] = True
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            checks.append(row)
            print(case["id"], row["passed"], flush=True)
        used = model.used_components
    passed = all(row["passed"] for row in checks) and used == set(manifest["components"])
    return {
        "passed": passed,
        "samples": len(checks),
        "checks": checks,
        "components_exercised": sorted(used),
        "capability_coverage": {
            name: {
                "status": "passed" if passed else "incomplete",
                "evidence": "Native full-response cases and all original component routes",
            }
            for name in manifest["capability_coverage"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = validate_conversion(args.artifact_path)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    else:
        with VelaFullCoreAI(args.artifact_path) as model:
            print(
                json.dumps(
                    model.system_one(
                        "Contact Alice at alice@example.com.",
                        {
                            "pii": {
                                "type": "span",
                                "criteria": {"EMAIL_ADDRESS": "an e-mail address"},
                            }
                        },
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
