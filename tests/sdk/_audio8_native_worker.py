import asyncio
import importlib.abc
import sys
import time
from pathlib import Path


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torchaudio", "transformers"}:
            raise RuntimeError("FORBIDDEN_RUNTIME_IMPORT:" + fullname)


sys.meta_path.insert(0, NoTorch())


async def main():
    import numpy as np
    import soundfile as sf
    from hugging_mac_sdk import ModelSdk, SpeechSynthesisRequest
    from hugging_mac_sdk.capabilities import SpeechSynthesis
    from hugging_mac_sdk.models.audio8_tts import register_audio8_tts
    from hugging_mac_sdk.schemas.transcription import AudioInput

    sdk = ModelSdk()
    d = register_audio8_tts(sdk.registry)
    start = time.perf_counter()
    async with await sdk.load(
        d.manifest.model_id,
        variant="0.1b-preview",
        runtime="coreai",
        options={"model_home": Path(sys.argv[1]), "device": "auto"},
    ) as h:
        print("NATIVE_LOAD", time.perf_counter() - start, flush=True)
        for label, request in [
            (
                "plain",
                SpeechSynthesisRequest(
                    text="你好, 这是一个测试。", max_new_tokens=48, do_sample=False
                ),
            ),
            (
                "repeat",
                SpeechSynthesisRequest(
                    text="你好, 这是一个测试。", max_new_tokens=48, do_sample=False
                ),
            ),
            (
                "reference",
                SpeechSynthesisRequest(
                    text="再说一次你好。",
                    reference_text="你好, 这是一个测试。",
                    reference_audio=AudioInput(path=Path(sys.argv[2]) / "audio8-native-plain.wav"),
                    max_new_tokens=32,
                    do_sample=False,
                ),
            ),
        ]:
            start = time.perf_counter()
            o = await h.require(SpeechSynthesis).synthesize(request)
            audio = np.frombuffer(o.audio, dtype="<f4")
            assert len(audio) and np.isfinite(audio).all() and abs(audio).max() > 0
            sf.write(Path(sys.argv[2]) / f"audio8-native-{label}.wav", audio, o.sample_rate)
            if label == "repeat":
                first, _ = sf.read(Path(sys.argv[2]) / "audio8-native-plain.wav", dtype="float32")
                assert np.max(np.abs(first - audio)) < 1e-3
            print(
                "NATIVE_SYNTH",
                label,
                time.perf_counter() - start,
                o.duration_seconds,
                o.generated_tokens,
                "torch_imported",
                "torch" in sys.modules,
                flush=True,
            )


asyncio.run(main())
