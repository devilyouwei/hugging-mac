# Qwen3 ASR

## Purpose

Qwen3 ASR transcribes multilingual speech with an Apple-oriented Core ML runtime.

## Support

- Model ID: `qwen/qwen3-asr`
- Variants: `0.6b` (default)
- Runtimes: `coreml` (default)
- Capabilities: `speech-transcription` through `SpeechTranscription`
- Platform: Apple Silicon macOS
- Optional dependency set: `asr`

## Prepare resources

Download the Core ML model and its required support files before loading it.

## Example

```python
import asyncio

from hugging_mac_sdk import AudioInput, ModelSdk, TranscriptionRequest
from hugging_mac_sdk.capabilities import SpeechTranscription
from hugging_mac_sdk.models.qwen3_asr import register_qwen3_asr


async def main() -> None:
    sdk = ModelSdk()
    register_qwen3_asr(sdk.registry)
    model_id = "qwen/qwen3-asr"

    await sdk.resources.download_source(model_id, variant="0.6b")
    async with await sdk.load(model_id, runtime="coreml") as handle:
        transcriber = handle.require(SpeechTranscription)
        result = await transcriber.transcribe(
            TranscriptionRequest(
                audio=AudioInput(path="speech.wav"),
                max_new_tokens=256,
            )
        )
        print(result.text)


asyncio.run(main())
```

## Notes and limits

The public response contains transcript text and timing information. It does not
expose the model's internal generation state.

## License

The model is licensed under `Apache-2.0`.
