# Gemma 4

## Purpose

Gemma 4 provides local text and image-aware chat on Apple Silicon.

## Support

- Model ID: `google/gemma-4`
- Variants: `e4b` (default), `e2b`
- Runtimes: `mlx` (default)
- Capabilities: `chat` through `Chat`
- Platform: Apple Silicon macOS
- Optional dependency set: `llm`

## Prepare resources

Download the variant that fits the available unified memory. `e2b` is the smaller
choice; `e4b` is the default.

## Example

```python
import asyncio

from hugging_mac_sdk import ChatMessage, ChatRequest, ModelSdk
from hugging_mac_sdk.capabilities import Chat
from hugging_mac_sdk.models.gemma_4 import register_gemma_4


async def main() -> None:
    sdk = ModelSdk()
    register_gemma_4(sdk.registry)
    model_id = "google/gemma-4"

    await sdk.resources.download_source(model_id, variant="e4b")
    async with await sdk.load(model_id, variant="e4b", runtime="mlx") as handle:
        chat = handle.require(Chat)
        result = await chat.chat(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="user",
                        content="Explain unified memory in one paragraph.",
                    ),
                )
            )
        )
        print(result.message.content)


asyncio.run(main())
```

## Notes and limits

Use `chat.stream_chat(request)` for incremental output. Model size, prompt length,
and images all contribute to unified-memory use.

## License

The package manifest declares `Apache-2.0`. Review the upstream Gemma terms that
apply to the selected model files.
