# MiniMind-V 运行与实验记录

## 1. 记录边界

本文件只记录已经在当前仓库实际检查或运行的事实。README 中的训练规模、官方效果图和宣传描述，不自动视为本机复现实验结果。

本轮目标不是完整重训，而是建立一个可验证的最小闭环：

1. 代码能导入并通过语法检查。
2. 本地 SigLIP2 与 VLM 权重能加载。
3. 图片能变成 64 个视觉 token，并正确替换文本占位 embedding。
4. 模型能对 6 张样图生成与主体相关的短文本。
5. 训练入口的前置条件和失败原因能够被定位。

实验日期：2026-07-13。

## 2. 当前环境

| 项目 | 实测值 |
|---|---|
| 操作系统 | Windows，PowerShell |
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime | 11.8 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Transformers | 4.57.6 |
| PyArrow | 16.1.0 |
| Pillow | 11.3.0 |

README 推荐 Python 3.10。本机是共享的全局 Anaconda Python 3.12 环境，虽然推理可运行，但 `pip check` 报告多组与本项目无直接关系的全局依赖冲突；此外 `swanlab` 和 `wandb` 当前未安装。结论是：**当前环境足够完成核心推理验证，但不应当被描述为干净、可复现的完整训练环境。**

### 推荐的独立环境

```powershell
conda create -n minimind-v python=3.10 -y
conda activate minimind-v

# 根据本机 CUDA 版本从 PyTorch 官网选择对应命令，再安装项目依赖。
pip install torch torchvision
pip install -r requirements.txt
python -m pip check
```

如果只学习核心代码，最小依赖是 `torch`、`transformers==4.57.6`、`Pillow`、`pyarrow`。训练日志功能需要额外确保 `swanlab` 可导入；WebUI 需要 `gradio`。

### 环境变量

单卡推理不需要 `.env` 或自定义环境变量。

- `dataset/lm_dataset.py:15` 在代码内部设置 `TOKENIZERS_PARALLELISM=false`。
- DDP 的 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 由 `torchrun` 注入。
- 本轮没有设置 API key，也没有调用外部模型服务。

## 3. 本地资产核对

### 视觉编码器

目录：`model/siglip2-base-p32-256-ve/`

- `model.safetensors`：189,129,296 bytes。
- `config.json`：输入 256、patch size 32、hidden size 768。
- `preprocessor_config.json`：图片缩放与归一化配置。

该目录已被 `.gitignore` 忽略，避免把大权重推送到 GitHub。

### VLM 权重

目录：`out/`

- dense：`llm_768.pth`、`pretrain_vlm_768.pth`、`sft_vlm_768.pth`。
- MoE：对应的 `*_768_moe.pth`。
- 本轮端到端验证使用 `out/sft_vlm_768.pth`，文件大小 140,051,548 bytes。

`out/` 同样被忽略。原生 VLM checkpoint 不含 `vision_encoder.*`，运行时从独立 SigLIP2 目录补回冻结视觉塔。

### 数据

- `dataset/eval_images/`：6 张推理样图，均存在且可解码。
- `dataset/pretrain_i2t.parquet`：本地不存在。
- `dataset/sft_i2t.parquet`：本地不存在。

因此本轮可以验证推理，不能运行真实 Pretrain/SFT epoch。

### 论文

已下载并纳入 Git：

| 文件 | 页数 | 首页内容核验 |
|---|---:|---|
| `papers/2304.08485-Visual-Instruction-Tuning-LLaVA.pdf` | 25 | 标题为 Visual Instruction Tuning |
| `papers/2310.03744-Improved-Baselines-with-Visual-Instruction-Tuning-LLaVA-1.5.pdf` | 15 | 标题为 Improved Baselines with Visual Instruction Tuning |

MiniMind-V 本身没有单独的项目论文；这两篇是其架构路线最直接的参考论文。PDF 已通过 `pypdf` 提取首页文本，并用 PyMuPDF 渲染首页做可读性检查。

## 4. 验证 1：语法与导入

命令：

```powershell
python -m compileall -q eval_vlm.py model dataset trainer scripts
```

结果：`PASS`。

这只能证明 Python 文件能被编译，不能证明模型数值正确，也不能覆盖数据、权重和 GPU 路径。

## 5. 验证 2：核心张量与参数

验证过程：加载 dense SFT 权重和 SigLIP2，对第一张样图构造 prompt，检查视觉编码、投影、embedding 替换、logits 和 KV cache。

```text
device                         NVIDIA GeForce RTX 4060 Laptop GPU
model + weight load            0.897 s
missing keys                   208，全部属于 vision_encoder.*
unexpected keys                0
pixel_values                   [1, 3, 256, 256]
input_ids                      [1, 93]
image marker id=12             64 个
raw vision features            [1, 64, 768]
projected vision features      [1, 64, 768]
mixed hidden states            [1, 93, 768]
marker replacement allclose    True
logits                         [1, 1, 6400]
KV cache                       8 层
vision + projector             0.674 s（含首次 CUDA 预热）
```

参数量：

| 参数集合 | 实测 |
|---|---:|
| 模型总参数（含冻结 SigLIP2） | 159.647M |
| SigLIP2 视觉编码器 | 94.552M |
| `freeze_llm=2` 可训练参数 | 1.183M |
| `freeze_llm=1` 可训练参数 | 15.932M |
| `freeze_llm=0` 可训练参数 | 65.095M |

这组结果验证了三个关键代码约定：

1. 一个 256x256、P32 的图像产生 8x8=64 个 patch token。
2. Projector 不改变 token 数，只把视觉表示映射到 LLM hidden space。
3. SFT checkpoint 故意不保存视觉编码器，因此 `strict=False` 的 missing keys 应全部来自 `vision_encoder.*`。

## 6. 验证 3：Label Mask

构造一条最小对话：

```text
user: <image>\n图中是什么？
assistant: 图中是一只金毛犬。
```

结果：

```text
fixed sequence length          128
image marker count             64
supervised token count         17
supervised text                <think>...</think> + 回答 + <|im_end|>
```

`generate_labels` 只把 assistant 段写入 labels，其余位置保持 `-100`。17 不是固定常数；它取决于回答长度和 chat template。这里真正需要验证的是：用户问题、system prompt、图片占位和 padding 不参与 cross entropy。

## 7. 验证 4：端到端 CLI Demo

命令：

```powershell
python eval_vlm.py `
  --load_from model `
  --weight sft_vlm `
  --max_new_tokens 12 `
  --image_dir .\dataset\eval_images `
  --show_speed 1 `
  --device cuda
```

真实输出摘要：

| 图片 | 生成文本（12 token 截断） | 速度 |
|---|---|---:|
| golden dog + balloons | 这张图片中的主要物体是一只金毛寻回 | 15.19 token/s |
| rainbow umbrella | 这张图片中的主要物体是雨伞。雨 | 59.78 token/s |
| cherry blossom + bike | 这幅图中主要物体是摩托车，背景 | 54.07 token/s |
| yellow car | 这张图片中的主要物体是一辆黄色的运动 | 56.81 token/s |
| superhero rooftop | 这张图片中的主要物体是超级英雄 | 57.98 token/s |
| racecar drift | 这张图片中的主要物体是赛车，具体来说 | 56.76 token/s |

判定：最小推理闭环通过。六个回答都命中了图片的主要物体，足以证明模型、视觉输入和文本生成链路已接通。

不能据此声称：

- 模型在标准 VLM benchmark 上达到某个水平。
- 回答细节完全正确。
- 模型不存在幻觉。
- 当前速度能代表稳定吞吐。

首图速度低是预热效应。要做正式性能测试，应先 warmup，再固定 prompt 长度、输出长度、dtype、sampling、同步边界，并报告多轮均值和 P50/P95。

### 单图 Greedy 复测

第一张图、`do_sample=False`、生成 16 token：

```text
prompt tokens                  93
generated tokens               16
elapsed                        0.539 s
throughput                     29.68 token/s
peak CUDA allocated memory     336.8 MB
output                         这张图片中的主要物体是一只金毛寻回犬，它
```

这个计时包含视觉编码和 prefill。不同 CUDA 缓存状态会显著影响短样本结果，因此只作为本机 sanity check。

## 8. 验证 5：训练入口与失败定位

### 失败 A：从仓库根执行

```powershell
python trainer\train_sft_vlm.py --epochs 0 --from_weight none --device cpu
```

结果：失败。`trainer_utils.init_vlm_model` 默认使用 `../model`，该相对路径以当前工作目录为基准，解析到 `D:\github\model`，随后 Hugging Face 报 `HFValidationError`。

当前正确启动方式：

```powershell
Set-Location trainer
python train_sft_vlm.py ...
```

更稳健的代码修复应当以 `Path(__file__).resolve()` 计算默认路径，消除 cwd 依赖。

### 失败 B：从 `trainer/` 启动

```powershell
Set-Location trainer
python train_sft_vlm.py --epochs 0 --from_weight none --device cpu
```

结果：tokenizer、视觉模型和 VLM 均成功初始化，并打印：

```text
Model Params: 65.09M
Trainable Params: 15.932M
```

随后在 `VLMDataset` 打开 `../dataset/sft_i2t.parquet` 时抛 `FileNotFoundError`。这是预期的外部数据缺失，不是模型初始化错误。

## 9. 依赖问题与解决表

| 现象 | 根因 | 解决方式 |
|---|---|---|
| `HFValidationError: '../model'` | 从错误 cwd 启动训练 | 进入 `trainer/`，或把路径改成基于 `__file__` |
| `FileNotFoundError: sft_i2t.parquet` | 训练数据未下载 | 下载到 `dataset/` 或传 `--data_path` |
| `vision_encoder is None` | SigLIP2 路径不存在或版本不兼容 | 检查本地目录、权重完整性和 Transformers 版本 |
| `swanlab` 导入失败 | 可选追踪依赖未安装 | 不传 `--use_wandb`，或安装 `swanlab==0.6.12` |
| CUDA OOM | batch、序列或图片 token 太多 | 减 batch、开梯度累积、缩短序列、用 bf16/fp16 |
| 输出不稳定 | 默认 `do_sample=True` 且每图随机 seed | 调试时使用 greedy 或固定 seed |
| WebUI 找不到模型 | 只扫描 Transformers 目录 | 先运行 `convert_vlm.py` 或扩展加载原生 `.pth` |
| pip 依赖冲突 | 使用共享全局 Python | 新建 Python 3.10 conda 环境，不在全局环境硬修 |

## 10. 测试缺口

仓库没有 `tests/` 目录，也没有 pytest/unittest 配置。当前验证仍缺少：

1. `count_vision_proj` 的 marker 长度错误、多图数量错误测试。
2. `generate_labels` 在截断、多轮、空回答下的边界测试。
3. 在线视觉特征与预计算特征的一致性测试。
4. train step 的 loss finite、梯度非零和 checkpoint round-trip 测试。
5. 标准多模态 benchmark 与纯文本能力回归。

因此，当前最准确的结论是“推理链路与核心组件已验证”，不是“项目完整训练和模型质量已复现”。

## 11. 下一轮实验建议

优先级从高到低：

1. 实现 `tests/test_vision_fusion.py`，把视觉 token 对齐假设变成可执行契约。
2. 下载一个小型 parquet 子集，跑 1-10 个 SFT step，验证 loss、梯度和 checkpoint。
3. 固定两张不同图片使用同一问题，比较视觉特征距离和生成差异，排除纯文本捷径。
4. 加入 warmup 后的 100 次 decode benchmark，报告 tokens/s 与峰值显存。
5. 做 projector Linear vs MLP 的小型消融，记录收敛速度与验证集 loss。
