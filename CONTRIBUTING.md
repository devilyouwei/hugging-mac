# Contributing

Thank you for helping improve Hugging Mac. Read the project boundaries in
[ARCHITECTURE.md](ARCHITECTURE.md) and the README nearest to the module you plan
to change before starting implementation.

## Core rules

1. A model capability enters the standalone SDK before an application consumes
   it. Application code must not depend on runtime model objects.
2. A new model declares its identity, capabilities, variants, runtimes, files,
   sources, and license in its model package before it is registered.
3. Do not commit model weights, datasets, credentials, personal paths, local
   caches, or large generated outputs.
4. Keep public APIs strongly typed. Application API schemas and SDK schemas are
   separate contracts with an explicit mapping at the business-service boundary.
5. Run the relevant tests and repository checks before opening a change.

## Documentation ownership

Documentation has one owner based on its audience:

- `docs/` contains only user workflows and public SDK usage.
- Root-level Markdown contains project-wide technical boundaries and contribution
  policy.
- A module's implementation outline, invariants, file responsibilities, and
  standards live in the nearest module README.
- Model-specific implementation details live in that model package, while the
  common model-package standard lives in the models directory README.

Do not copy a model ID, variant list, runtime list, source, file inventory, or
other static fact into multiple technical files. Refer to the current
`model.yaml`; public model pages are checked against it automatically.

Public documentation may explain concepts such as variants, runtimes, artifacts,
and capabilities from the caller's perspective. It must not expose private class
relationships, internal artifact IDs, graph names, managed physical paths, or
conversion algorithms.

## Dependencies

Add runtime dependencies to the smallest applicable optional extra. Development
tools belong in a dependency group. A large dependency change must document:

- why existing dependencies cannot provide the capability;
- Apple Silicon and Python 3.12 compatibility;
- installation size and runtime-memory impact;
- licensing and model-distribution constraints.

## Architecture decisions

Use an ADR only for a difficult-to-reverse decision that affects more than one
module or changes a public contract. Add a root-level `ADR-NNNN-short-title.md`
with status, context, decision, alternatives, consequences, migration, and
verification. Small implementation details belong in code or the module README.

## Checks

Run the checks relevant to the change. The complete local suite is:

```bash
uv run ruff check .
uv run mypy packages/hugging_mac_sdk/src apps/web/backend/src
uv run pytest
uv run --group docs mkdocs build --strict
cd apps/web/frontend && npm run build
```
