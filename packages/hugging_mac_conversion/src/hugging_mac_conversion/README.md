# Model Conversion Workbench

模型作者工具，与 `hugging_mac_sdk` 平级。两个包互不导入、没有依赖关系；转换包独立维护
已有的 Core AI authoring/runtime，负责显式转换、验证、制品整理和 Hub 发布。

## 目录与依赖方向

```text
packages/
├── hugging_mac_sdk/                 # SDK 自有依赖、源码和运行时
│   ├── pyproject.toml
│   └── src/hugging_mac_sdk/
└── hugging_mac_conversion/          # 独立转换包、锁文件和环境
    ├── pyproject.toml               # 转换依赖、dev group、CLI 与打包
    ├── uv.lock
    ├── .python-version
    ├── .venv/                      # uv 管理，Git 忽略
    ├── tests/                      # 转换流程与评测基础设施的离线测试
    └── src/hugging_mac_conversion/
        ├── __main__.py             # CLI 参数、模型/变体选择
        ├── registry.py             # 显式组合模型定义
        ├── models/
        │   ├── vela/
        │   │   ├── model.yaml      # 模型事实、完整能力、各变体配置
        │   │   ├── definition.py   # 配置与适配器的组合
        │   │   ├── converter.py    # 模型专属加载与图适配
        │   │   ├── evaluation/     # config.yaml、cases/、runner.py
        │   │   └── readme.md
        │   └── d3/                 # 相同分工，独立配置与实现
        └── shared/                 # 配置合同、导出、存储、Hub、验证、评测
```

依赖方向：`CLI → registry → model definition → shared contracts`；CLI 将模型定义传入
`shared.pipeline`。`shared` 不导入模型包或 registry，不按模型名分支。模型不互相导入。
只创建实际使用的文件，不为每个模型复制 Core ML / Core AI 通用导出实现。

静态事实唯一保存在每个模型的 `model.yaml`：来源、发布目标、完整能力合同、变体、
图接口和诊断所需文件。`definition.py` 只组合配置与 callable；`converter.py` 只处理
模型算法。新增兼容变体只扩展该模型的 `variants`，不新增模型目录。不同架构或任务合同
不强行共用一个模型。只有完成验证的变体才可作为可用模型发布。

## uv 环境

从仓库根目录执行：

```bash
uv sync --project packages/hugging_mac_conversion --locked
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert list
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert probe vela --variant 3-fast --runtime coreml
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert probe d3 --variant edge --runtime coreai
```

省略 `--variant` 时使用模型 YAML 的默认变体；未知变体在下载前报错。`--revision` 选择
上游分支/tag/commit，运行时解析为不可变 commit，代码与权重使用同一快照。
本地快照下载完成后，可用 `probe --offline --revision <source-commit>` 校验文件摘要后离线转换。

转换依赖由独立包的 `pyproject.toml` 管理，开发工具放在 `dev` group，不使用 requirements。
工具项目从主 workspace 排除，单独锁定和同步：Web 的 SDK extras 固定 Torch 2.7，
Apple Core AI 转换器需要 Torch >=2.8，不能共用运行环境。主 workspace 的 `uv sync
--all-packages` 仍按原来的应用依赖工作。更新转换依赖时在工具项目执行 `uv lock`。

Apple 官方工具为 `coremltools`、`coreai-torch`、`coreai-core`。转换包所需版本的原生包及 Torch
安装到独立转换环境；支持情况以当前锁定版本及 Apple Silicon/Python 3.12 实测为准。
模型和编译缓存可能占用数 GB，转换峰值内存随图和权重增长。源码许可证保留在模型目录，
模型许可证随制品复制；权重不进入 Python wheel。
当前 Core AI 工具链锁定 Torch 2.14 / coreai-torch 0.4.3 / coreai-core 1.0.0b3，
与已有 D3 Core AI 仓库的配方版本对齐；转换器仍会提示 Torch 2.14 超过其声明的验证范围。
Core AI 导出不额外执行 `program.optimize()`；测试结论以实际制品和运行后端为准。

## 原始模型、转换制品与评测

三类数据全部默认保存在项目根目录的 `tmp/`，由 Git 忽略，不进入 Python wheel：

```text
tmp/
├── originals/<model>/<variant>/<source-commit>/  # 原始代码、配置、权重及 .source.json
├── conversions/<model>/<variant>/<runtime>/<run>/
└── evaluations/<model>/<variant>/<run>/         # report.json + report.md
```

原始模型是项目内的真实文件，不是指向全局缓存的软链接。不同上游 commit 分开保存，
`.source.json` 记录仓库、commit、下载范围和文件 SHA-256。已迁移的旧缓存标记为
`cached-partial`；`probe` 仅下载该诊断所需文件，记录为 `probe`。默认 `fetch` 完整下载，
不套用 probe 的文件过滤；完成后才记录 `full`。需要完整快照时执行：

```bash
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert fetch vela --variant 3-fast
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert fetch d3 --variant edge
```

`fetch` 和 `probe` 都支持 `--source-root`，默认 `tmp/originals`。同一变体和 commit 复用
同一个目录；`fetch --probe-files` 可显式只准备诊断文件。已有全局缓存不会被删除。

每个模型的 `evaluation/` 独立维护 `config.yaml`（用例文件、容差）、`cases/*.json`
（输入和预期结果）、`runner.py`（上游预处理、原始模型推理、原生输入适配）。用例可用
`variants` 限定变体，省略时适用于该模型全部变体。模型配置仍由 `model.yaml` 统一维护。
共享库只负责执行、指标和报告，不包含模型名分支。

```bash
# 对比某次转换与其对应 commit 的本地原始模型；--run 可以重复，比较多个制品。
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate vela --variant 3-fast --run tmp/conversions/vela/3-fast/coreml/<run>
# 只运行原始模型基线：revision 必须是已下载的不可变 commit。
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert evaluate d3 --variant edge --revision <source-commit>
```

Core AI 可用 `--coreai-device cpu` 显式选择 CPU（需要系统运行时支持）；
报告记录设备选择及 `USE_OS_COREAI`。

评测不下载模型；先校验原始文件摘要、制品摘要及来源 commit，再运行原始模型与候选后端。
不同 commit 的制品不能混在一次对比中。`--output` 默认 `tmp/evaluations`，每次创建独立
JSON 与 Markdown 报告，保留用例摘要、工具版本、逐例概率、决策、预期结果、误差及推理耗时；不会修改转换
bundle。耗时是单次观测，不是稳定性能基准。

当前用例是 **diagnostic**：Vela 覆盖 choice/noul，D3 覆盖 choice/noul/score。
数值容差与决策一致性决定转换对比是否通过，预期答案正确性另行记录；两边同样答错不代表
任务质量通过。超出当前诊断接口的用例记为未评测，其他完整能力仍标记未测试。
这些报告不替代完整模型验收，也不能解除诊断制品的发布限制。

## 临时制品与发布

```text
tmp/conversions/<model>/<variant>/<runtime>/<run>/
├── build.json                  # 选中变体配置、source commit、工具版本、状态与摘要
├── model.mlpackage/            # 或 model.aimodel/
├── assets/                     # 该变体声明的 tokenizer/config/license
└── README.md
```

默认从当前目录向上定位仓库根，子目录运行也写入同一个 `tmp/conversions`。此目录被
根 `.gitignore` 中的 `tmp/` 忽略；`--output` 可显式选择其他位置。每次运行创建唯一目录，
不覆盖旧产物；失败保留报告。目录就是待检查/上传的 bundle，不额外复制到源码目录。
不同变体分别使用自身发布目标，同一目标下的 `coreml/`、`coreai/` 分开上传。

```bash
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert verify-bundle tmp/conversions/<model>/<variant>/<runtime>/<run>
uv run --project packages/hugging_mac_conversion --locked hugging-mac-convert push tmp/conversions/<model>/<variant>/<runtime>/<run>
# 真正上传时在上一条命令后添加 --execute；默认只显示计划。
```

身份来自 `hf auth login` 或 `HF_TOKEN`，不写入配置。发布前检查文件摘要、选中变体配置、
全部上游能力及原生制品验收，并在实际上传前重新运行原生验收。转换包中的验收回调通过模型配置
的 `bundle_validator` 注入。SDK 公共能力验收独立放在主工作区的 `tests/sdk/`。
回调返回 `passed`、至少两个任务样例的 `samples`，以及逐项带 `status` / `evidence` 的
`capability_coverage`。

**当前 `probe` 是内部计算图诊断，不是完整模型转换。** 诊断产物不能升级为完整模型或
进入发布流程。最终必须保留上游完整输入输出能力；未测试不能视为不支持，只有可复现
的转换障碍才可作为能力缺失的依据。完整 SDK 接入与任务验证完成前不注册可用模型。
模型各自的实现状态由对应 `readme.md` 维护。

旧 `artifacts/conversions` 制品已保留并迁移到新的模型/变体目录。旧报告缺少版本化配置
合同，不能直接发布；重新构建后使用新报告。源码下的旧隔离环境与 requirements 已移除。

## 多图模型发布

Zipformer 的流式 encoder、decoder 和 joiner 作为一组发布。它使用工作台的
Core ML 导出和制品目录，并在模型目录内验收完整的流式命中链路。命令见
[Zipformer 转换说明](models/zipformer/readme.md)。

## 验证

```bash
cd packages/hugging_mac_conversion
uv run --locked pytest
uv run --locked ruff check src tests
uv run --locked mypy src/hugging_mac_conversion
```

主 workspace 的 `pytest` 继续验证 SDK 和应用；转换包的测试使用自己的环境和配置。
模型任务评测仍在源码中随转换包分发，`tests/` 只保存开发回归测试。

离线测试覆盖模型/变体隔离、配置校验、依赖方向、临时路径、注入式流程、失败报告与发布
门禁。模型权重、网络请求和原生框架推理只出现在显式诊断或 smoke 中。
