# Getting Started

This guide installs Hugging Mac from the repository and starts the local Studio.

## Requirements

- An Apple Silicon Mac
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 20.19+ or 22.12+

## Install the workspace

From the repository root:

```bash
uv python install 3.12
uv sync --all-packages
```

Install the frontend packages:

```bash
cd apps/web/frontend
npm ci
```

## Start the Studio

Start the local API from the repository root:

```bash
uv run hugging-mac-web
```

In another terminal, start the frontend:

```bash
cd apps/web/frontend
npm run dev
```

Open <http://127.0.0.1:5173/>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

## Prepare your first model

1. Open **Models** in the Studio navigation.
2. Expand a model card and choose a variant.
3. Download an artifact for the runtime you want to use.
4. If the runtime is provided through local conversion, download its source and
   then select **Convert**.
5. When the runtime reports that all files are ready, select **Load instance**.
6. Open a compatible App or Game and run a request.

Downloads and conversions are always explicit. Opening an App or loading a model
does not silently fetch files from the internet.

## Optional model dependencies

The base workspace includes the project applications. When installing only the
SDK package, select the extras needed by your workload:

```bash
uv sync --package hugging-mac-sdk --extra yolo
uv sync --package hugging-mac-sdk --extra asr
uv sync --package hugging-mac-sdk --extra tts
uv sync --package hugging-mac-sdk --extra vad
uv sync --package hugging-mac-sdk --extra llm
```

## Next steps

- Learn how resource actions and runtime readiness work in the
  [Model Library guide](model-library.md).
- Explore local experiences in the [Studio guide](studio.md).
- Embed models in Python with the [SDK guide](sdk/README.md).
