# MiniMind-V 深度代码库学习日志

## 总体研究进展

### 项目目标

从第一性原理彻底掌握 MiniMind-V 当前代码库，形成可验证、可交接、可用于面试训练的仓库地图、原理说明、主链路追踪、核心模块笔记和实验记录。

### 研究方向

- 以当前源码为真相源，验证模型架构、数据流、训练策略和推理链路。
- 追踪一个最小图片问答示例，从 `eval_vlm.py` 到视觉编码、token 替换、Transformer 生成。
- 区分项目宣传口径、代码事实、运行事实和尚未验证的假设。
- 设计递进练习与面试问题，帮助学习者从“会复述”进阶到“能修改和扩展”。

### 已完成工作

- 已阅读上一阶段日志 `experiment journal 20260617.md`。
- 已确认当前分支为 `main`，远端为用户个人仓库 `https://github.com/cupkk/MiniMind-V.git`，工作区初始干净。
- 已确认本地保留 SigLIP2 视觉编码器、dense/MoE 权重和两篇 LLaVA 参考论文。
- 已确认上一阶段曾用 CPU 跑通 6 张图片的短生成验证。
- 已生成并审阅 `PROJECT_MAP.md`、`FIRST_PRINCIPLES.md`、`EXECUTION_TRACE.md`、`MODULE_NOTES.md`。
- 已在 RTX 4060 Laptop GPU 上重新跑通 dense SFT 权重的 6 图 CLI 推理。
- 已完成核心 shape、视觉 token 替换、KV cache、冻结参数量与 label mask 的组件验证。
- 已核验两篇本地 LLaVA PDF 的页数、首页文字与渲染可读性。

### 当前关键发现

- 项目规模较小，核心源代码集中在 `model/`、`dataset/`、`trainer/`、`eval_vlm.py` 和 `scripts/`。
- 仓库没有独立的 `tests/` 目录或自动化测试框架；需要以语法检查、组件级断言和端到端推理作为验证闭环。
- 本地运行环境是全局 Anaconda Python 3.12，而 README 推荐 Python 3.10；运行结论必须注明环境差异。

### 当前阻塞项

- 完整训练依赖大规模 parquet 数据与 GPU，本轮不计划执行完整训练。
- 现有全局环境曾出现 pip 依赖冲突，不能把“当前可推理”外推为“所有训练依赖均健康”。
- 本地缺少 `pretrain_i2t.parquet` 和 `sft_i2t.parquet`，无法运行真实训练 epoch。

### 下一步

1. 完成文档一致性检查与 Git 发布。
2. 进入单题面试模式，按回答逐项指出理解漏洞。
3. 后续按练习 1-3 逐步增加可观测性、严格融合检查和视觉特征缓存。

## 2026-07-13 更新

### 初始定向

- 操作：检查文件清单、Git 状态、远端和最近提交。
- 结果：当前 `main` 跟踪 `origin/main`，最新提交为 `4de1e47 Add MiniMind-V study notes`。
- 决策：保留现有学习材料，但本轮新文档必须重新绑定到具体文件、类、函数和实际命令，不直接复用旧摘要作为结论。

### 源码全局审计

- 已读源码：`model/model_minimind.py`、`model/model_vlm.py`、`dataset/lm_dataset.py`、`trainer/trainer_utils.py`、两份训练入口、`eval_vlm.py`、`scripts/convert_vlm.py`、`scripts/web_demo_vlm.py`、tokenizer 配置与依赖文件。
- 代码确认的模型事实：
  - LLM 主干是 decoder-only Causal LM，使用 RMSNorm、GQA、RoPE、SwiGLU，可选 MoE。
  - VLM 融合发生在 embedding 层：64 个 `<|image_pad|>` embedding 被 SigLIP2 patch 特征经 MLP Projector 后的结果替换。
  - tokenizer 配置确认 `<|image_pad|>` 的 token id 为 12。
  - 视觉编码器全程冻结，但每次训练 forward 仍需执行视觉编码计算。
- 工程发现：
  - 仓库没有 `tests/` 或 pytest/unittest 自动化测试。
  - `VLMDataset` 会把完整 parquet 组装成 Arrow Table，超大数据集会占用显著主存。
  - 训练工具默认路径使用 `../model`、`../out`、`../dataset`，实际依赖从 `trainer/` 目录启动；从仓库根执行 `python trainer/train_*.py` 会解析到错误目录。
  - `eval_vlm.init_model` 用 `'model' in args.load_from` 判断原生权重模式，任意包含 `model` 的 Transformers 路径可能被误判。
  - WebUI 只把当前消息送入模型，UI 中的 `history` 没有进入 prompt，因此不是真正多轮上下文。
  - WebUI 对 chat template 结果按字符数截断，不是按 token 数截断，存在截断特殊 token 的风险。
- 决策：上述内容将在学习文档中作为“隐含假设与潜在坑点”明确列出，不把 README 宣称等同于代码保证。

### 五份学习文档

- 新增：`PROJECT_MAP.md`，覆盖项目定位、目录职责、入口、依赖、配置、测试现状、阅读顺序和最小实现边界。
- 新增：`FIRST_PRINCIPLES.md`，从输入/输出/中间状态出发解释 patch token、Projector、embedding 替换、Causal LM、GQA、RoPE、MoE、KV cache、正确性和性能指标。
- 新增：`EXECUTION_TRACE.md`，从 `eval_vlm.py` 追踪到图片预处理、视觉编码、Projector、token 融合、8 层 Transformer、生成与输出。
- 新增：`MODULE_NOTES.md`，深挖 9 个模块，建立论文概念到代码的映射，并给出 3 个递进练习和面试题库。
- 新增：`EXPERIMENT_LOG.md`，区分运行事实、代码事实和未完成项，记录本轮全部关键命令与输出。

### 运行环境与静态检查

- 环境：Python 3.12.3、PyTorch 2.7.1+cu118、Transformers 4.57.6、RTX 4060 Laptop GPU。
- 操作：执行 `python -m compileall -q eval_vlm.py model dataset trainer scripts`。
- 结果：通过。
- 依赖审计：`pip check` 返回多组全局环境冲突；`swanlab`、`wandb` 当前未安装。核心推理依赖可用，完整训练环境不应宣称健康。
- 决策：不在共享 Anaconda 环境中强行升级/降级依赖，文档推荐独立 Python 3.10 conda 环境。

### 核心组件验证

- 输入：第一张金毛犬样图和固定中文描述问题。
- 结果：`pixel_values=[1,3,256,256]`，`input_ids=[1,93]`，id 12 的 image marker 为 64 个。
- 结果：SigLIP2 输出 `[1,64,768]`，Projector 输出 `[1,64,768]`，融合后 hidden states 为 `[1,93,768]`。
- 断言：marker 位置与 Projector 输出 `allclose=True`。
- 输出：单步 logits `[1,1,6400]`，KV cache 为 8 层。
- 参数：总参数 159.647M，视觉编码器 94.552M；三种 freeze 模式可训练参数分别为 1.183M、15.932M、65.095M。
- 问题与修复：首次验证脚本漏传静态方法 `get_image_embeddings` 的 `vision_model` 参数，按源码签名补传 `model.vision_encoder` 后通过；该问题属于临时验证脚本，不是项目源码缺陷。

### Label Mask 验证

- 构造 user 图片问题和 assistant 回答“图中是一只金毛犬。”。
- 结果：128 长度序列含 64 个 image marker，17 个 token 参与监督。
- 解释：17 包含模板生成的空 think 段、回答和 `<|im_end|>`；数量不是固定值，核心约束是 user/system/image/padding 位置为 `-100`。

### 端到端推理

- 命令：`python eval_vlm.py --load_from model --weight sft_vlm --max_new_tokens 12 --image_dir .\dataset\eval_images --show_speed 1 --device cuda`。
- 结果：6 张图全部生成与主体相关的短描述，识别到金毛犬、雨伞、摩托车、黄色汽车、超级英雄和赛车。
- 速度：首图 15.19 token/s，后 5 图 54.07-59.78 token/s；首图受模型与 CUDA 预热影响。
- 单图 greedy 复测：16 token 耗时 0.539 秒，29.68 token/s，峰值 allocated memory 336.8MB。
- 决策：这些结果仅证明最小闭环，不能写成标准 benchmark 成绩。

### 训练失败路径验证

- 从仓库根运行 `trainer/train_sft_vlm.py`：默认 `../model` 路径解析错误，抛 `HFValidationError`。
- 从 `trainer/` 运行：模型成功初始化并打印 15.932M 可训练参数，随后因缺少 `../dataset/sft_i2t.parquet` 抛 `FileNotFoundError`。
- 建议：短期从 `trainer/` 启动；长期把默认路径改为基于 `Path(__file__).resolve()`。

### 论文核验

- `2304.08485-Visual-Instruction-Tuning-LLaVA.pdf`：25 页，首页标题和正文提取正常。
- `2310.03744-Improved-Baselines-with-Visual-Instruction-Tuning-LLaVA-1.5.pdf`：15 页，首页标题、雷达图和架构图渲染正常。
- 本地缺少 Poppler `pdftoppm`，改用 PyMuPDF 渲染首页，未影响核验。

### 收尾检查

- 更新 `.gitignore`：增加通用 `__pycache__/` 和 `*.py[cod]`，避免验证命令产生的缓存进入提交。
- 清理：在逐项解析并确认路径位于 `D:\github\MiniMind-V` 后，删除根目录及 `dataset/`、`scripts/`、`trainer/` 下的 `__pycache__`，并删除 PDF 首页临时渲染目录 `tmp/`。
- 文档完整性：五份目标文档全部存在；代码围栏数量均为偶数；共包含 10 个 Mermaid 图块。
- Git 检查：`git diff --check` 通过；待提交范围仅为五份学习文档、当日日志和 `.gitignore`。
- 保留：`model/siglip2-base-p32-256-ve/`、`out/` 继续忽略，两篇论文 PDF 已在上一阶段提交，不重复修改。

### Git 发布

- 目标远端：用户个人仓库 `https://github.com/cupkk/MiniMind-V.git`。
- 目标分支：`main`。
- 文档提交：`453c8b8 docs: add first-principles MiniMind-V study guide`。
- 推送结果：`4de1e47..453c8b8 main -> main`。
- 远端核验：第一次推送后，`git rev-parse HEAD` 与 `git ls-remote --heads origin main` 均为 `453c8b8ad83394fa16ecbda5550ddadac76be411`。
- 下一步：完成本条发布日志的收尾提交后，再次核验最终本地/远端哈希一致和工作区干净。
