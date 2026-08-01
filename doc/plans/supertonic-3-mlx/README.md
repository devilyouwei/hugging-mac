# Supertonic 3 MLX 实现计划

状态：`Deferred`

本计划暂缓执行。当前优先实现 VLM；完成 VLM 的 SDK 与应用闭环后，再评估并接入本模型。

## 暂定目标

- 模型：`Supertone/supertonic-3`；
- 首选 runtime：MLX；
- 目标设备：Apple Silicon GPU；
- 能力：多语言文本转语音、预置音色、流式输出；
- 定位：低延迟本地 TTS，优先补足 Audio8 ONNX INT4 无法有效使用 Apple GPU/ANE 的问题。

MLX 路径以 Apple GPU 和统一内存为主，不声明 Neural Engine 支持。只有在独立的 Core ML 或
ExecuTorch 产物通过正确性、稳定性和性能验证后，才能增加 ANE runtime。

## 计划目录

正式实现时创建：

```text
packages/hugging_mac_sdk/src/hugging_mac_sdk/models/supertonic_3/
├── __init__.py
├── config.py
├── definition.py
├── instance.py
├── mlx.py
├── model.yaml
├── resources.py
└── utils/
    ├── __init__.py
    ├── engine.py
    ├── text.py
    ├── types.py
    └── voices.py
```

在实现完成前，不在 `models/` 下创建占位包，不注册 model definition，也不加入模型目录测试白名单。

## 实现阶段

1. 审查上游模型、MLX 实现、模型权重和 OpenRAIL 许可证，固定 repo、revision 和文件摘要。
2. 建立项目级 MLX runtime 依赖与设备能力探测，确认不会影响现有 PyTorch、ONNX 和 Core ML runtime。
3. 按模型 SDK 约定实现 definition、resource、instance 和 `SpeechSynthesis` adapter。
4. 支持预置音色、语言、语速、随机种子、流式 chunk、取消和显式 unload。
5. 在 text-to-speech Demo 增加模型选项，但保持应用层不直接依赖 MLX 对象。
6. 增加单元、资源、生命周期、真实音频 smoke test 和端到端应用测试。

## 性能验收

在项目目标 Apple Silicon 机器上，与 Kokoro PyTorch MPS 和 Audio8 ONNX INT4 使用相同文本比较：

- 冷加载时间与峰值统一内存；
- 首个音频 chunk 延迟；
- 完整生成耗时和 real-time factor；
- 短句、长文本及连续十次生成的稳定性；
- unload 后内存回收；
- 中文和英文的可懂度、重复、漏读及异常噪声。

默认 runtime 必须同时满足：输出有效、连续运行不崩溃、速度显著优于 CPU 基线。不能仅根据 provider
名称或模型能够加载就声明 GPU/ANE 支持。

## 待确认事项

- 采用 MLX Community 权重还是在仓库转换脚本中从官方权重生成；
- 上游 MLX 实现的维护活跃度和 API 稳定性；
- OpenRAIL 条款是否符合项目预期分发方式；
- 中文文本规范化、分句和多音字策略；
- 是否需要 Core ML/ExecuTorch 作为第二 runtime；
- 若 Supertonic 3 的中文质量未达到要求，是否改用 Qwen3-TTS 0.6B MLX。

## 参考

- [Supertonic 官方仓库](https://github.com/supertone-inc/supertonic)
- [Supertonic 3 MLX 模型](https://huggingface.co/mlx-community/supertonic-3)
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
