# Audio8 TTS Preview

## Purpose

Audio8 TTS generates multilingual speech and supports reference-audio voice
cloning on compatible runtimes.

## Support

- Model ID: `audio8/audio8-tts-preview`
- Variants: `0.1b-preview` (default), `0.6b-preview`
- Runtimes: `pytorch` (default), `mlx`, `coreai`
- Capabilities: `speech-synthesis` through `SpeechSynthesis`
- Compatibility: both variants use CPU by default with PyTorch; `0.6b-preview` also supports MLX; `0.1b-preview` supports native Core AI
- Platform: MLX requires Apple Silicon
- Optional dependency set: `tts`

## Prepare resources

The Core AI example below uses the 0.1B variant. Choose the same variant
and runtime for preparation and loading. Preconverted graphs and the shared tokenizer are published in
[hugging-mac/audio8-tts-0.1b-coreai](https://huggingface.co/hugging-mac/audio8-tts-0.1b-coreai).
If large-file transfer stalls on your network, set `HF_HUB_DISABLE_XET=1`
before starting Python to use Hugging Face's standard HTTP download path.

## Example

For inference, PyTorch and Transformers are not needed. From the repository root:

```bash
python3.12 -m venv .venv-audio8-coreai
.venv-audio8-coreai/bin/pip install -e './packages/hugging_mac_sdk[coreai]' \
  tokenizers scipy soundfile
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
Slow AR (Falcon-H1), fast acoustic codebook prediction, reference-audio encoding,
and waveform decoding all run in Core AI. NumPy handles prompt arrays and
sampling; the Rust tokenizer handles text, and SoundFile/SciPy load and resample
reference audio. Persistent KV, convolution, and recurrent states stay in native
sessions and reset between requests. Prompt tokens are ingested sequentially.
The default `device="auto"` lets the runtime select accelerators; `cpu` is also
available. Native conversion does not guarantee ANE placement or a speedup over
the older hybrid build. Loading a current SDK requires downloading or converting
the native build; existing hybrid files are kept separately.
The shared tokenizer remains reusable across runtimes. Loading never downloads
or converts a model. Pass `overwrite=True` to explicitly replace a conversion.

Local conversion uses a separate environment with `coreai-torch==0.4.2`,
`torch==2.8.0`, `torchaudio==2.8.0`, `transformers==4.57.5`, and `safetensors`;
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
