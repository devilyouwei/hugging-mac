# DeepFilterNet3

## Purpose

DeepFilterNet3 removes background noise from recorded speech.

## Support

- Model ID: `deepfilternet/deepfilternet3`
- Variants: `default` (default)
- Runtimes: `coreml` (default)
- Capabilities: `speech-enhancement` through `SpeechEnhancement`
- Platform: Apple Silicon macOS
- Optional dependency set: `asr`

## Prepare resources

Download the Core ML model before loading it.

## Example

```python
import asyncio
from pathlib import Path

from hugging_mac_sdk import AudioInput, ModelSdk, SpeechEnhancementRequest
from hugging_mac_sdk.capabilities import SpeechEnhancement
from hugging_mac_sdk.models.deepfilternet3 import register_deepfilternet3


async def main() -> None:
    sdk = ModelSdk()
    register_deepfilternet3(sdk.registry)
    model_id = "deepfilternet/deepfilternet3"

    await sdk.resources.download_source(model_id)
    async with await sdk.load(model_id, runtime="coreml") as handle:
        enhancer = handle.require(SpeechEnhancement)
        result = await enhancer.enhance_speech(
            SpeechEnhancementRequest(audio=AudioInput(path="noisy.wav"))
        )
        Path("enhanced.wav").write_bytes(result.audio)


asyncio.run(main())
```

## Notes and limits

Input may be mono or multi-channel encoded audio. Set `output_sample_rate` on the
request when a specific output rate is required.

## License

The model is licensed under `Apache-2.0`.
