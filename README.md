<p align="center">
  <img src="apps/web/frontend/public/hugging-mac-icon.png" alt="hugging-mac" width="144">
</p>

# hugging-mac

**Run neural network models on your Mac.**

hugging-mac is a local model playground built for Apple Silicon. It brings classic
CNN, RNN, YOLO, and CLIP models together with modern LLMs and VLMs in one extensible
SDK and web platform.

> The project is in active early development. The model SDK, web platform, and first
> YOLOv8 object-detection app are available now.

## Highlights

- A standalone Python SDK with consistent model lifecycle APIs
- Multiple isolated instances of the same model
- Pluggable model capabilities such as `detect`, `chat`, and `transcribe`
- Runtime selection for PyTorch MPS, Core ML, MLX, and future backends
- Model download, resource management, and format conversion
- A FastAPI backend and a Vue 3 frontend
- A foundation for demos, evaluation, training, and benchmarking

## Requirements

- An Apple Silicon Mac
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 20.19+ or 22.12+ for the web frontend

## Quick start

Install the Python workspace and YOLO dependencies:

```bash
uv python install 3.12
uv sync --all-packages --extra yolo
```

Start the API:

```bash
uv run hugging-mac-web
```

In another terminal, start the frontend:

```bash
cd apps/web/frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173/>. The API documentation is available at
<http://127.0.0.1:8000/docs>.

In the Object Detection app, download a YOLOv8 variant and run it with MPS, or
convert it to Core ML for an Apple-optimized runtime.

## Documentation

The architecture and design documentation starts at
[doc/README.md](doc/README.md). The design documents are currently written in
Chinese.

Serve the documentation locally with Material for MkDocs and Mermaid support:

```bash
uv run --group docs mkdocs serve --dev-addr 127.0.0.1:8001
```

Then open <http://127.0.0.1:8001/>.

Key documents:

- [Architecture](doc/01-architecture.md)
- [Model SDK](doc/02-model-sdk.md)
- [Model catalog and capabilities](doc/03-model-catalog.md)
- [Model resource management](doc/04-resource-management.md)
- [Demo platform](doc/05-demo-platform.md)
- [Evaluation](doc/06-evaluation.md)
- [Training](doc/07-training.md)
- [Frontend, backend, and API](doc/08-frontend-backend.md)
- [Platform backend](doc/17-platform-backend.md)
- [Vue and object detection](doc/18-web-frontend-and-object-detection.md)
- [Engineering conventions](doc/10-engineering.md)
- [Roadmap](doc/11-roadmap.md)

## Repository layout

```text
hugging-mac/
├── apps/
│   └── web/
│       ├── backend/              # FastAPI platform and app APIs
│       └── frontend/             # Vue platform and app pages
├── packages/
│   └── hugging_mac_sdk/          # Standalone model SDK
├── configs/                      # Versioned configuration
├── data/                         # Local datasets, ignored by default
├── artifacts/                    # Generated outputs, ignored by default
├── models/                       # Local model cache, ignored by default
├── tests/
├── doc/
├── pyproject.toml
└── uv.lock
```

## Development

Install development and documentation dependencies:

```bash
uv sync --all-packages --extra yolo --group dev --group docs
```

Run the checks:

```bash
uv run ruff check .
uv run mypy packages apps/web/backend
uv run pytest
uv run --group docs mkdocs build --strict
cd apps/web/frontend && npm run build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before contributing.

## License

hugging-mac is released under the [MIT License](LICENSE).

Model weights, datasets, and third-party dependencies retain their own licenses.
Review their terms before use. Ultralytics YOLO software and models, for example,
have separate licensing requirements.
