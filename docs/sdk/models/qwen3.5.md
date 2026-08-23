# Qwen3.5

## Purpose

Qwen3.5 provides local chat and vision-language prompts through one chat API.

## Support

- Model ID: `qwen/qwen3.5`
- Variants: `9b` (default), `4b`, `2b`
- Runtimes: `mlx` (default)
- Capabilities: `chat`, `vision-language-generation` through `Chat`
- Platform: Apple Silicon macOS
- Optional dependency set: `llm`

## Prepare resources

Choose a size based on available unified memory. The example uses the 4B variant.

## Example

```python
import asyncio

from hugging_mac_sdk import ChatImage, ChatMessage, ChatRequest, ModelSdk
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.models.qwen3_5 import register_qwen3_5


async def main() -> None:
    sdk = ModelSdk()
    register_qwen3_5(sdk.registry)
    model_id = "qwen/qwen3.5"

    await sdk.resources.download_source(model_id, variant="4b")
    async with await sdk.load(model_id, variant="4b", runtime="mlx") as handle:
        chat = handle.require(Chat)
        result = await chat.chat(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="user",
                        content="Describe this image.",
                        images=(ChatImage(path="photo.jpg"),),
                    ),
                )
            )
        )
        print(result.message.content)


asyncio.run(main())
```

## Notes and limits

Images can be supplied from a local path or URL. Use the smallest variant that
meets the task when memory pressure is a concern.

## License

The model is licensed under `Apache-2.0`.
