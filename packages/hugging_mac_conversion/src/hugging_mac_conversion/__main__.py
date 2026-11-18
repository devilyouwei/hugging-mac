"""CLI composition only: choose a model/variant and call shared services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import MODELS, get_model
from .shared.artifacts import (
    default_evaluation_root,
    default_output_root,
    default_source_root,
    source_directory,
)
from .shared.evaluation.engine import evaluate
from .shared.hub import download_source, resolve_revision
from .shared.pipeline import build_probe, publish
from .shared.validation import verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Mac model conversion workbench")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List models and their configured variants")
    probe = commands.add_parser("probe", help="Diagnose graph conversion; not a full release")
    probe.add_argument("model", choices=sorted(MODELS))
    probe.add_argument("--variant", help="Defaults to the model's configured default variant")
    probe.add_argument("--runtime", choices=["coreml", "coreai"], required=True)
    probe.add_argument("--coreai-device", choices=["auto", "cpu"], default="auto")
    probe.add_argument("--revision", default="main")
    probe.add_argument(
        "--offline", action="store_true", help="Use a verified local immutable --revision"
    )
    probe.add_argument("--output", type=Path, help="Defaults to <repository>/tmp/conversions")
    probe.add_argument("--source-root", type=Path, help="Defaults to <repository>/tmp/originals")
    fetch = commands.add_parser("fetch", help="Save an original model snapshot inside the project")
    fetch.add_argument("model", choices=sorted(MODELS))
    fetch.add_argument("--variant")
    fetch.add_argument("--revision", default="main")
    fetch.add_argument("--source-root", type=Path)
    fetch.add_argument(
        "--probe-files", action="store_true", help="Fetch only diagnostic dependencies"
    )
    evaluation = commands.add_parser(
        "evaluate", help="Compare model-owned cases against local artifacts"
    )
    evaluation.add_argument("model", choices=sorted(MODELS))
    evaluation.add_argument("--variant")
    evaluation.add_argument(
        "--run", type=Path, action="append", default=[], help="Repeat to compare multiple artifacts"
    )
    evaluation.add_argument(
        "--revision", help="Immutable source commit; required for reference-only evaluation"
    )
    evaluation.add_argument("--source-root", type=Path)
    evaluation.add_argument("--coreai-device", choices=["auto", "cpu"], default="auto")
    evaluation.add_argument("--output", type=Path, help="Defaults to <repository>/tmp/evaluations")
    verify = commands.add_parser(
        "verify-bundle", help="Run the owning model's native task-parity checks"
    )
    verify.add_argument("run", type=Path)
    push = commands.add_parser("push", help="Inspect or publish a complete validated bundle")
    push.add_argument("run", type=Path)
    push.add_argument("--execute", action="store_true", help="Upload (default: dry-run)")
    args = parser.parse_args()
    try:
        if args.command == "list":
            print(json.dumps([d.config.model_dump(mode="json") for d in MODELS.values()], indent=2))
        elif args.command == "probe":
            definition = get_model(args.model)
            variant = args.variant or definition.config.default_variant
            definition.config.variant(variant)
            build_probe(
                definition,
                variant,
                args.runtime,
                args.output or default_output_root(),
                args.revision,
                args.source_root,
                args.offline,
                args.coreai_device,
            )
        elif args.command == "fetch":
            definition = get_model(args.model)
            variant = args.variant or definition.config.default_variant
            config = definition.config.variant(variant)
            revision = resolve_revision(config, args.revision)
            destination = source_directory(
                args.source_root or default_source_root(), args.model, variant, revision
            )
            print(download_source(config, revision, destination, probe=args.probe_files))
        elif args.command == "evaluate":
            definition = get_model(args.model)
            variant = args.variant or definition.config.default_variant
            result = evaluate(
                definition,
                variant,
                args.run,
                args.source_root or default_source_root(),
                args.output or default_evaluation_root(),
                args.revision,
                args.coreai_device,
            )
            report = json.loads((result / "report.json").read_text())
            if report["status"] in {"failed", "error", "incomplete"}:
                raise SystemExit(1)
        else:
            report = json.loads((args.run / "build.json").read_text())
            if report.get("schema_version") != 1:
                raise ValueError(
                    "Legacy build report; rebuild with the current model configuration"
                )
            definition = get_model(report["model"])
            if args.command == "verify-bundle":
                verify_bundle(definition, args.run, report)
            else:
                print(publish(definition, args.run, report, dry_run=not args.execute))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
