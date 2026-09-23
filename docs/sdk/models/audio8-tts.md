# Audio8 TTS Preview

## Purpose

Audio8 TTS generates multilingual speech and supports reference-audio voice
cloning on compatible runtimes.

## Support

- Model ID: `audio8/audio8-tts-preview`
- Variants: `0.1b-preview` (default), `0.6b-preview`
- Runtimes: `pytorch` (default), `mlx`, `coreai`
- Capabilities: `speech-synthesis` through `SpeechSynthesis`
- Compatibility: both variants use CPU by default with PyTorch; `0.6b-preview` also supports MLX; `0.1b-preview` supports hybrid Core AI
- Platform: MLX requires Apple Silicon
- Optional dependency set: `tts`

## Prepare resources

The Core AI example below uses the 0.1B variant. Choose the same variant
and runtime for preparation and loading. Preconverted graphs, retained CPU
weights, and the shared tokenizer are published in
[hugging-mac/audio8-tts-0.1b-coreai](https://huggingface.co/hugging-mac/audio8-tts-0.1b-coreai).
If large-file transfer stalls on your network, set `HF_HUB_DISABLE_XET=1`
before starting Python to use Hugging Face's standard HTTP download path.

## Example

Use a separate Python 3.12 environment so the tested PyTorch version
does not conflict with the application's existing model dependencies. From the
repository root:

```bash
python3.12 -m venv .venv-audio8-coreai
.venv-audio8-coreai/bin/pip install -e './packages/hugging_mac_sdk[coreai]' \
  'torch==2.8.0' 'torchaudio==2.8.0' \
  'transformers==4.57.5' safetensors soundfile
```

Download the published files explicitly, then load the same
variant with `runtime="coreai"`:

```python
import asyncio
from pathlib import Path
from hugging_mac_sdk import ModelSdk, SpeechSynthesisRequest
from hugging_mac_sdk.capabilities import SpeechSynthesis
from hugging_mac_sdk.models.audio8_tts import register_audio8_tts

async def main():
    sdk = ModelSdk()
    register_audio8_tts(sdk.registry)
    model_id = "audio8/audio8-tts-preview"
    options = {"model_home": Path("models")}
    await sdk.resources.download_source(model_id, variant="0.1b-preview",
                                        options=options | {"runtime": "coreai"})
    async with await sdk.load(model_id, variant="0.1b-preview", runtime="coreai",
                             options=options) as handle:
        result = await handle.require(SpeechSynthesis).synthesize(
            SpeechSynthesisRequest(text="你好，这是一个语音合成测试。", max_new_tokens=128)
        )
        Path("audio8-coreai.f32le").write_bytes(result.audio)

asyncio.run(main())
```

Core AI execution requires Apple Silicon and compatible Apple runtime support.
This is a hybrid pipeline: acoustic codebook prediction and waveform decoding
run in Core AI; text/semantic generation and reference-audio encoding run in
PyTorch on CPU. It retains the PyTorch dependencies and source weights, so
conversion increases disk use. It does not promise faster synthesis.
The shared tokenizer remains reusable across runtimes. Loading never downloads
or converts a model. Pass `overwrite=True` to explicitly replace a conversion.

Local conversion remains available: additionally install `coreai-torch==0.4.2`,
download resources with `runtime="pytorch"`, and call
`sdk.resources.convert(model_id, ArtifactFormat.COREAI, variant="0.1b-preview", options=options)`
after importing `ArtifactFormat` from `hugging_mac_sdk`. Download and conversion
publish the same declared files at the same managed path. Use `overwrite=True`
to replace an existing download or repair an incomplete directory.

## Notes and limits

`reference_audio` and `reference_text` must be provided together. PyTorch uses
CPU by default because it currently performs better for both variants. Callers
can explicitly request MPS; after an MPS availability, loading, or inference
failure, the engine retries once on CPU. MLX GPU execution is limited to
`0.6b-preview`.

## License

The default 0.1B variant uses `audio8-community-license-v1.0`; the 0.6B variant
retains its `Apache-2.0` terms. Review the selected upstream model license before
redistribution. The 0.1B source is
[Edge0/Audio8-TTS-Preview-0.1b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.1b).
