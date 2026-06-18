# MiniMind-V 学习复现指南

这份文档面向“准备用 MiniMind-V 作为多模态大模型实习项目写进简历”的学习目标。建议和源码里的中文注释一起看：先跑通，再读代码，再把原理讲顺。

## 1. 项目一句话

MiniMind-V 是一个极简 VLM（Vision-Language Model）项目：用冻结的 SigLIP2 视觉编码器提取图片 patch 特征，再用一个两层 MLP Projector 把视觉特征投影到 MiniMind LLM 的 hidden space，最后把这些视觉 token 替换到文本里的 `<|image_pad|>` 占位符位置，让原本只会处理文本 token 的自回归语言模型也能“读图回答”。

面试里可以这样讲：

> 我复现了一个轻量级 VLM 项目，核心思路类似 LLaVA：冻结视觉编码器，用 MLP projector 做跨模态对齐，把图像 patch token 注入到 LLM 的 token embedding 序列中，再通过图文 SFT 训练模型完成图片描述和视觉问答。

## 2. 技术路线总览

MiniMind-V 的链路可以拆成 5 步：

1. 图片输入：PIL Image 经过 `SiglipImageProcessor` resize / normalize，变成 `pixel_values`。
2. 视觉编码：`SiglipVisionModel` 输出 64 个 patch token，每个 token 维度 768。
3. 跨模态投影：`MMVisionProjector` 执行 `LayerNorm -> Linear -> GELU -> Linear`，把视觉 token 映射到 LLM hidden size。
4. 占位符替换：文本 prompt 中的 `<image>` 会展开成 64 个 `<|image_pad|>`；模型 forward 时用投影后的视觉 token 替换这些位置的文本 embedding。
5. 自回归生成：替换后的 embedding 序列进入 MiniMind Transformer，后续和普通 Causal LM 一样预测下一个 token。

可以把它理解为：

```text
图片 -> SigLIP2 -> 64 个视觉 token -> MLP Projector -> 64 个 LLM 空间 token
文本 -> tokenizer -> 含 64 个 <|image_pad|> 的 token 序列 -> embedding
然后：用视觉 token 替换 image pad embedding -> MiniMind LLM -> 生成回答
```

## 3. 你必须掌握的源码地图

### `model/model_vlm.py`

这是最重要的文件。要重点看：

- `VLMConfig`：定义图像占位 token、图像 token 数量、视觉特征维度。
- `MMVisionProjector`：跨模态 projector，把视觉特征变成 LLM 能接收的 hidden state。
- `MiniMindVLM.get_vision_model`：加载并冻结 SigLIP2。
- `MiniMindVLM.count_vision_proj`：真正把 `<|image_pad|>` 的 embedding 替换成视觉 embedding。
- `MiniMindVLM.forward`：首步注入图像特征，后续生成依赖 KV cache。

面试高频问题：

- 为什么视觉编码器冻结？
- 为什么要 projector？
- 为什么是 64 个 image token？
- `<image>` 和 `<|image_pad|>` 有什么区别？
- 为什么只在 `start_pos == 0` 时注入图像？

### `dataset/lm_dataset.py`

这是训练数据如何变成模型输入的文件。要重点看：

- `create_chat_prompt`：把数据里的 `<image>` 替换为 64 个 `<|image_pad|>`。
- `generate_labels`：只让 assistant 回复参与 loss，user/system prompt 的 label 是 `-100`。
- `__getitem__`：从 parquet 读出 conversations 和 image bytes，并把图片转成 SigLIP 输入。

面试高频问题：

- 为什么 prompt 部分不计算 loss？
- 多模态样本在 parquet 里怎么存？
- 训练时图像和文本怎么对齐？

### `trainer/trainer_utils.py`

这是训练工程细节。要重点看：

- `init_vlm_model`：权重加载和冻结策略。
- `freeze_llm=0/1/2`：全参训练、首尾层训练、只训 projector。
- `vlm_checkpoint`：保存推理权重和 resume 训练状态。
- `vlm_collate_fn`：把单条样本合成 batch。

面试高频问题：

- Pretrain 和 SFT 的冻结策略为什么不同？
- 为什么 SFT 只解冻 LLM 首尾层？
- 为什么保存权重时不保存 vision encoder？

### `trainer/train_pretrain_vlm.py`

Pretrain 阶段默认只训练 projector，目标是先建立“视觉 token -> 语言空间”的基础对齐。

建议你这样理解：

> 这一步像给 LLM 学一门外语的词典，只让翻译器先学会把图片特征翻译成 LLM 能读的 token 表示，不动 LLM 本体。

### `trainer/train_sft_vlm.py`

SFT 阶段训练 projector 和 LLM 首尾层，目标是让模型学会按指令回答图片相关问题。

建议你这样理解：

> SFT 不只是让 projector 对齐，还要让 LLM 的输入层适应视觉 token，让输出层适应视觉问答的回答风格。中间层保留原有语言能力。

### `eval_vlm.py`

这是命令行推理入口。要重点看：

- `init_model`：区分原生 PyTorch 权重和 Transformers 格式模型。
- `MiniMindVLM.image2tensor`：把图片转成 processor 输出。
- `prompt.replace('<image>', ...)`：推理时必须和训练时一样展开 image pad。
- `model.generate(..., pixel_values=pixel_values)`：把图像传给模型。

## 4. 复现路线

### 第一步：安装依赖

建议新建虚拟环境后执行：

```bash
pip install -r requirements.txt
```

如果要用 GPU，先确认 PyTorch 和 CUDA 匹配：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### 第二步：下载视觉编码器和权重

README 推荐 ModelScope：

```bash
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
modelscope download --model gongjy/minimind-3v-pytorch --local_dir ./out
```

如果只想快速推理，可以下载已发布权重；如果要从训练开始，至少需要 `llm_768.pth` 作为语言底座。

### 第三步：先跑推理

```bash
python eval_vlm.py --load_from model --weight sft_vlm
```

它会自动读取 `dataset/eval_images/` 下的测试图片并生成描述。

### 第四步：再跑最小训练

快速复现可以跳过 Pretrain，直接跑 SFT：

```bash
python trainer/train_sft_vlm.py --epochs 1 --from_weight llm --batch_size 1 --save_interval 100
```

但前提是你已经把 `sft_i2t.parquet` 放到 `dataset/` 下。

如果你想按完整路线复现：

```bash
python trainer/train_pretrain_vlm.py --epochs 1 --from_weight llm
python trainer/train_sft_vlm.py --epochs 1 --from_weight pretrain_vlm
```

## 5. 底层原理拆解

### 为什么 VLM 可以用 LLM 改出来？

LLM 本质上处理的是 token 序列。文本 token 是离散 id，再查 embedding 表变成向量。图片不能直接变成文本 token，但可以先由视觉编码器变成一串连续向量。只要这些连续向量被投影到 LLM hidden space，就可以和文本 embedding 拼在同一条序列里，让 Transformer 统一建模。

### 为什么需要 MLP Projector？

视觉 encoder 的 hidden state 和 LLM embedding 虽然维度可能相同，但语义空间不一样。Projector 的作用是学习一个跨模态映射，把“视觉特征语言”翻译成 LLM 更容易理解的“内部表示语言”。

### 为什么是 64 个视觉 token？

当前使用的 SigLIP2 是 P32，输入图像固定为 256×256。patch size 是 32，所以一张图被切成：

```text
256 / 32 = 8
8 * 8 = 64 patches
```

每个 patch 输出一个 token，因此一张图对应 64 个视觉 token。

### 为什么训练时冻结视觉编码器？

冻结 SigLIP2 有三个好处：

- 省显存和训练成本。
- 小数据训练不容易破坏视觉 encoder 原有能力。
- 项目目标是教学和低成本复现，训练 projector 已经能展示 VLM 核心思路。

### 为什么 SFT 只解冻首尾层？

MiniMind-V 的 LLM 只有约 64M。如果 SFT 阶段全参训练，图文任务梯度可能覆盖原来的语言能力。只解冻首尾层是折中方案：

- 首层负责接收并融合视觉 token。
- 末层影响输出分布和回答风格。
- 中间层保留语言模型预训练学到的通用能力。

## 6. 简历写法

可以写成一条偏工程复现的项目经历：

```text
复现并解析轻量级视觉语言模型 MiniMind-V：基于冻结 SigLIP2 视觉编码器与 MiniMind Causal LM，实现图像 patch token 经 MLP Projector 注入 LLM token embedding 序列的多模态对齐流程；完成图文 parquet 数据加载、assistant-only loss 构造、Pretrain/SFT 两阶段训练脚本阅读与注释，并整理推理、权重转换和 WebUI 调用链路。
```

更偏面试可讲的版本：

```text
MiniMind-V 多模态模型复现：参考 LLaVA 的视觉指令微调思路，使用 SigLIP2 提取 64 个图像 patch token，通过两层 MLP 投影到 LLM hidden space，并替换文本序列中的 image pad embedding；理解并复现 projector-only pretrain 与 projector+LLM 首尾层 SFT 的低成本训练策略，完成模型推理和源码中文注释。
```

## 7. 面试问答准备

### Q1：MiniMind-V 和 LLaVA 的关系是什么？

MiniMind-V 思路上参考 LLaVA：都是把视觉 encoder 输出投影到 LLM hidden space，再让 LLM 进行图文生成。区别是 MiniMind-V 更小、更教学化，语言底座是 MiniMind，视觉 encoder 使用 SigLIP2，projector 是简单两层 MLP，训练策略更强调低成本和冻结参数。

### Q2：图像到底是怎么进入 LLM 的？

图像先被 SigLIP2 编码成 64 个 patch token，再通过 `vision_proj` 投影到 LLM hidden size。文本 prompt 中的 `<image>` 被替换成 64 个 `<|image_pad|>`，模型在 forward 时找到这些 token 的 embedding 位置，并用视觉 token 替换掉它们。之后 Transformer 看到的就是一条混合了视觉 token 和文本 token 的序列。

### Q3：为什么 labels 里有很多 `-100`？

`-100` 是 PyTorch cross entropy 的 ignore index。训练聊天模型时，我们不希望模型学习预测 system 和 user prompt，只希望学习 assistant 应该如何回答，所以 prompt 部分 label 设为 `-100`，assistant 回复部分保留真实 token id。

### Q4：Pretrain 和 SFT 分别学什么？

Pretrain 学基础图文对齐，默认只训练 projector，让图片特征进入语言空间。SFT 学按用户指令回答图片问题，默认训练 projector 和 LLM 首尾层，使模型既能融合视觉信息，又尽量保留原有语言能力。

### Q5：这个项目有哪些局限？

- 固定 256×256 输入，细粒度视觉信息有限。
- 单图为主，多图/视频/定位能力不是重点。
- LLM 底座很小，语言推理和细节描述能力受限。
- 视觉 encoder 冻结，无法针对任务域深度适配。
- 评估主要是样例展示，不是严格 benchmark。

## 8. 后续学习建议

建议按这个顺序学习：

1. 先读 `MiniMind-V 学习复现指南.md` 和 README 的“模型细节”部分。
2. 跑 `eval_vlm.py`，确认模型能对图片生成回答。
3. 读 `model/model_vlm.py`，画出图像 token 替换流程。
4. 读 `dataset/lm_dataset.py`，确认 labels 如何构造。
5. 读 `trainer/trainer_utils.py`，理解冻结策略。
6. 读 LLaVA 和 LLaVA-1.5 论文，理解 projector 与 visual instruction tuning 的来源。
7. 准备一页自己的项目复盘：动机、方法、实现、实验、局限、改进。
