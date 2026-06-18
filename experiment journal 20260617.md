# MiniMind-V 复现与面试准备研究日志

## 总体研究进展

### 项目目标

本工作区用于复现和学习 `jingyaogong/minimind-v` 项目，目标是把它整理成一个可以写入多模态大模型岗位实习简历的项目经历。当前优先级是：先复现源码和运行链路，再理解代码与底层原理，最后形成面试可讲的项目拆解、简历表述和问答准备。

### 研究方向

- 理解 MiniMind-V 的整体技术路线：视觉编码器、MLP 投影层、语言模型、训练阶段、推理流程。
- 阅读并注释核心源码：数据加载、模型结构、预训练、监督微调、推理与转换脚本。
- 确认项目是否有配套论文；如果有，下载到本地并做后续阅读。
- 形成新手友好的中文学习材料，帮助应对面试官对代码实现、模型结构、训练目标和复现细节的追问。

### 已完成工作

- 发现本地 `D:\github\MiniMind-V` 初始只有 `.git`，没有源码内容。
- 原远端为 `https://github.com/cupkk/MiniMind-V.git`，该远端没有可拉取分支。
- 根据用户提供链接，将 `origin` 改为 `https://github.com/jingyaogong/minimind-v.git`，并检出 `origin/master` 到本地 `master`。
- 当前源码目录已恢复，包含 `dataset/`、`model/`、`trainer/`、`scripts/`、`eval_vlm.py`、`README.md` 等。

### 当前关键发现

- 这是一个很小型的 VLM 训练/推理项目，适合用来学习多模态大模型从视觉特征到语言模型输入的核心链路。
- 仓库内暂未发现 PDF 或论文文件，后续需要从 README、GitHub 页面和外部搜索确认是否存在论文链接。
- 当前首要代码阅读范围是所有 Python 源码与 README，而不是只看训练入口。

### 当前阻塞项

- 尚未安装依赖、下载模型权重或数据集，暂未做可运行复现验证。
- 尚未确认 Hugging Face 数据集/权重是否需要额外网络下载或认证。
- 尚未确认项目是否有正式论文。

### 下一步

1. 完整阅读 README、核心 Python 源码与配置文件。
2. 梳理项目模块图、训练流程图和推理调用链。
3. 查找论文或技术报告链接，若存在则下载到本地。
4. 选择必须掌握的核心源码文件，加入新手友好的详细中文注释。
5. 做最小复现检查，输出环境搭建、运行命令、可能踩坑和面试准备材料。

## 2026-06-17 更新

### 13:40 左右：恢复项目源码

- 操作：检查本地工作区，发现只有 `.git`，没有源码文件。
- 操作：检查 `git remote -v`，发现原远端为 `cupkk/MiniMind-V.git`。
- 结果：`git fetch origin` 后没有拉到有效分支，本地显示 `origin/main [gone]`。
- 决策：以用户明确给出的 `jingyaogong/minimind-v` 为真相源，修改 `origin` 并拉取 `master`。
- 命令结果：`git checkout -B master origin/master` 成功，本地分支 `master` 已跟踪 `origin/master`。
- 需要保留：当前远端已被修正为用户目标仓库，不要再切回旧的 `cupkk/MiniMind-V.git`。
- 下一位 agent：请先读本日志，再从 README 和 Python 源码继续做深度拆解。

### 13:43 左右：完成第一轮源码与论文状态盘点

- 阅读文件：`README.md`、`model/model_minimind.py`、`model/model_vlm.py`、`dataset/lm_dataset.py`、`trainer/train_pretrain_vlm.py`、`trainer/train_sft_vlm.py`、`trainer/trainer_utils.py`、`eval_vlm.py`、`scripts/convert_vlm.py`、`scripts/web_demo_vlm.py`、`requirements.txt`、`model/model_init.md`。
- 核心理解：
  - MiniMind-V 不是完全重新发明 VLM，而是在 MiniMind Causal LM 上增加 `SiglipVisionModel` 视觉编码器和 `MMVisionProjector`。
  - 输入文本中的 `<image>` 会被替换成 64 个 `<|image_pad|>`，与 SigLIP2 P32 在 256×256 图像上输出的 8×8=64 个视觉 patch token 对齐。
  - 视觉特征经过 `LayerNorm -> Linear -> GELU -> Linear` 投影到 LLM hidden size，然后替换掉 image pad token 的文本 embedding。
  - Pretrain 阶段默认 `freeze_llm=2`，只训练 `vision_proj`，用于基础图文对齐。
  - SFT 阶段默认 `freeze_llm=1`，训练 `vision_proj` 与 LLM 首尾层，保护中间层语言能力。
  - 推理脚本 `eval_vlm.py` 支持原生 PyTorch 权重和 Transformers 格式模型两种加载方式。
- 论文状态：
  - 仓库内没有 MiniMind-V 自己的论文 PDF。
  - README 的引用是 GitHub 仓库引用格式：`MiniMind-V: Train a Tiny VLM from Scratch`。
  - README 明确参考了 LLaVA 和 LLaVA-1.5；已下载对应 arXiv PDF 到本地：
    - `papers/2304.08485-Visual-Instruction-Tuning-LLaVA.pdf`
    - `papers/2310.03744-Improved-Baselines-with-Visual-Instruction-Tuning-LLaVA-1.5.pdf`
- 决策：注释优先覆盖新手理解门槛最高的文件：`model/model_vlm.py`、`dataset/lm_dataset.py`、`trainer/trainer_utils.py`、`eval_vlm.py`，必要时再补充 `model/model_minimind.py` 中的 Transformer 基础结构注释。

### 13:50 左右：补充中文注释、下载资源并完成短推理验证

- 修改文件：
  - `model/model_vlm.py`：补充 VLM 配置、视觉编码器冻结、MLP Projector、图像 token 替换、KV cache 首步注入、loss 计算等中文注释。
  - `dataset/lm_dataset.py`：补充 parquet 数据读取、`<image>` 展开、chat template、assistant-only labels、图片 bytes 处理等中文注释。
  - `trainer/trainer_utils.py`：补充参数统计、学习率、DDP 初始化、冻结策略、checkpoint/resume、collate 和 skip sampler 注释。
  - `trainer/train_pretrain_vlm.py`：补充 Pretrain step、loss、梯度累积、保存权重、只训练 projector 的说明。
  - `trainer/train_sft_vlm.py`：补充 SFT 与 Pretrain 的差异、低学习率、首尾层训练策略说明。
  - `eval_vlm.py`：补充原生权重/Transformers 权重加载、推理时图像占位符展开、采样参数说明。
  - `MiniMind-V 学习复现指南.md`：新增面向新手和面试准备的中文复现指南。
- 下载文件：
  - `model/siglip2-base-p32-256-ve/`：已通过 ModelScope 下载 SigLIP2 视觉编码器。
  - `out/`：已通过 ModelScope 下载 `gongjy/minimind-3v-pytorch` 发布权重，包括 dense 和 MoE 的 LLM、Pretrain、SFT 权重。
  - `papers/2304.08485-Visual-Instruction-Tuning-LLaVA.pdf`
  - `papers/2310.03744-Improved-Baselines-with-Visual-Instruction-Tuning-LLaVA-1.5.pdf`
- 环境变更：
  - 当前 Python 为 3.12.3，而 README 推荐 Python 3.10。
  - 为了先跑通推理，安装了 `transformers==4.57.6`、`datasets==3.6.0`、`gradio==5.49.1`。
  - pip 输出显示当前全局 Anaconda 环境存在若干依赖冲突；后续正式训练建议新建 Python 3.10 虚拟环境，不要继续污染全局环境。
- 验证结果：
  - `python -m compileall model dataset trainer scripts eval_vlm.py` 已通过。
  - `python eval_vlm.py --load_from model --weight sft_vlm --max_new_tokens 32 --image_dir ./dataset/eval_images --show_speed 0 --device cpu` 已通过，6 张测试图均产生输出。
  - 因 `max_new_tokens=32`，输出被刻意截断；该验证只证明加载和推理链路可运行，不代表完整效果评估。
- 当前问题：
  - CPU 推理可跑但速度慢；建议后续在 CUDA 环境中运行。
  - 全局 Python 3.12 环境和项目推荐环境不一致，训练阶段可能遇到额外兼容性问题。
- 下一位 agent：
  - 若要继续训练，请先创建 Python 3.10 虚拟环境并重新安装依赖。
  - 若只是学习和面试准备，优先阅读 `MiniMind-V 学习复现指南.md` 与已注释的 6 个核心文件。

### 后续更新：根据用户目标改为“项目掌握与面试准备”

- 用户澄清：当前不需要真正训练复现，核心目标是彻底理解项目流程、代码逻辑、数学原理、面试八股与简历表达。
- 新增文件：`MiniMind-V 项目掌握与面试手册.md`。
- 文档内容：
  - 项目 30 秒版本和 2 分钟版本口述稿。
  - Mermaid 项目整体流程图、分层架构图、代码地图、图像注入流程图、训练阶段图、MoE 图、知识树。
  - `model_vlm.py`、`lm_dataset.py`、`trainer_utils.py` 等核心代码逻辑拆解。
  - Causal LM、Embedding、Self-Attention、RoPE、MoE、SigLIP、Projector 等数学与理论基础。
  - 面试官高频追问与浅显标准答案。
  - 一轮完整模拟面试对话。
  - 可直接放进简历的“任务背景 → 方法/实现 → 实验结果/产出”三段式版本。
- 决策：后续学习应以 `MiniMind-V 项目掌握与面试手册.md` 为主，`MiniMind-V 学习复现指南.md` 作为补充；真实训练不是当前优先级。
