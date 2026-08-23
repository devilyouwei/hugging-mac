# Nemotron 3.5 ASR Streaming 0.6B

## Purpose

Nemotron 3.5 ASR provides multilingual, stateful streaming transcription.

## Support

- Model ID: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Variants: `latin-560ms`, `latin-1120ms`, `latin-2240ms`, `latin-4480ms`,
  `multilingual-560ms`, `multilingual-1120ms`, `multilingual-2240ms` (default),
  `multilingual-4480ms`
- Runtimes: `coreml` (default)
- Capabilities: `speech-transcription`, `streaming-speech-transcription` through
  `SpeechTranscription` and `StreamingSpeechTranscription`
- Platform: Apple Silicon macOS
- Optional dependency set: `asr`

## Prepare resources

Download the exact latency/language variant that the streaming session will use.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import AudioInput, ModelSdk, StreamingTranscriptionRequest
from hugging_mac_sdk.capabilities import StreamingSpeechTranscription
from hugging_mac_sdk.models.nemotron_3_5_asr import register_nemotron_3_5_asr


async def main() -> None:
    sdk = ModelSdk()
    register_nemotron_3_5_asr(sdk.registry)
    model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    variant = "multilingual-2240ms"

    await sdk.resources.download_source(model_id, variant=variant)
    async with await sdk.load(model_id, variant=variant, runtime="coreml") as handle:
        transcriber = handle.require(StreamingSpeechTranscription)
        session = await transcriber.start_stream()
        partial = await transcriber.transcribe_stream(
            StreamingTranscriptionRequest(
                session_id=session.session_id,
                audio=AudioInput(data=Path("chunk.wav").read_bytes()),
            )
        )
        final = await transcriber.finish_stream(session.session_id)
        print(partial.delta, final.text)


asyncio.run(main())
```

## Notes and limits

Use consecutive chunks from one audio stream and always finish or cancel the
session. Shorter windows favor latency; longer windows favor throughput and
context.

## License

The model is licensed under `OpenMDW-1.1`.
