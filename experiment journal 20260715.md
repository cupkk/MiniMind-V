# MiniMind-V 原理讲义与面试准备日志

## 总体研究进展

### 项目目标

将已有的代码导向型材料重写为一份面向初学者的统一原理讲义，确保学习者能从图像、向量、token、Transformer、训练损失和推理效率的因果链理解 MiniMind-V，并能在 2026-07-16 与浙江大学李环老师交流时准确回答专业追问。

### 已完成工作

- 已读取 `experiment journal 20260713.md`，继承此前源码审计、张量验证、端到端推理和 Git 发布结论。
- 已确认当前 `main` 与 `origin/main` 同步，工作区初始干净。
- 已核对李环老师的浙江大学官方主页、个人主页和公开论文页面。
- 已确认其当前公开研究重点为 data-centric、resource-efficient、scalable AI，具体包含大模型/多模态大模型高效推理、模型轻量化、数据准备、时空与多模态数据。

### 关键决策

- 新文档不按源码目录或函数组织，而按“图片如何被模型理解”这一条因果链组织。
- 公式必须同时给出符号含义、直觉解释和 MiniMind-V 中的真实数字，避免只有推导没有理解。
- 面试问题分为基础原理、项目设计、正确性与效率、导师方向四类，但统一放在一个文档中。
- 对导师问题只做基于公开研究方向的高概率预测，明确不声称知道真实面试题。
- 项目成果口径保持诚实：已经跑通推理和组件验证，但没有完整重训或标准 benchmark。

### 当前阻塞项

- 无文档编写阻塞。
- 完整训练数据仍未下载，因此不能新增真实训练曲线或 benchmark 结果。

### 下一步

1. 完成统一原理与面试讲义。
2. 检查专业词解释、公式符号、Mermaid 图和口述答案的一致性。
3. 提交并推送个人仓库 `main`。
4. 进入单题模拟面试，根据用户回答逐项补漏洞。

## 2026-07-15 更新

### 导师方向核验

- 浙江大学官方主页将李环老师的长期方向概括为“资源高效、以数据为中心的人工智能方法和应用”，包括人工智能数据准备、大模型高效推理与部署、时空大数据和模型轻量化。
- 个人主页进一步列出 Efficient AI 方向：轻量模型、联邦学习、高效 LLM 推理与微调，并扩展到非结构化和多模态数据。
- 代表性相关工作包括 ACL 2024 `Draft & Verify` 的无损自推测解码、EMNLP 2025 `SpecVLM` 的验证器引导视觉 token 裁剪，以及 2026 年 Efficient LVLM Inference 资料库。
- 对 MiniMind-V 的面试准备重点据此调整为：视觉 token 为什么昂贵、prefill/decode 有何差异、KV cache 如何增长、怎样在不明显损害质量的前提下减少计算，以及数据质量如何影响对齐。

### 统一讲义编写

- 新增 `MiniMind-V 原理与李环老师面试通关讲义.md`，作为不依赖其他代码文档的独立入口。
- 规模：约 2.6 万字符、677 行、12 个大章节、5 个 Mermaid 图、22 组专业问答。
- 组织方式：先用“图片向量翻译”建立直觉，再补 Tensor、shape、token、embedding、Linear、Softmax 和 Cross Entropy，随后解释视觉 patch、SigLIP2、Projector、embedding 替换、Self-Attention 和两阶段训练。
- 公式均绑定项目真实数字，包括 256/P32 得到 64 patch、Projector 1.183M 参数、KV cache 近似内存和 93-token 实测链路。
- 推理部分新增 prefill/decode、KV cache、GQA、视觉 token 成本、量化、LoRA、视觉 token 裁剪和推测解码的浅显解释。
- 面试部分根据导师公开方向加入数据质量、效率测量、反事实视觉验证、视频扩展、Speculative Decoding 和导师匹配回答。
- 口径：明确 6 图输出只是 sanity check；没有完整训练和标准 benchmark；不声称从零发明架构。

### 初步完整性检查

- Markdown 围栏 22 个且成对闭合。
- Mermaid 图 5 个。
- 面试问答 22 组。
- `git diff --check` 通过，当前仅新增统一讲义和当日日志。

### 技术与可读性复核

- 对照源码确认：hidden size 768、8 层、8 个 Q head、4 个 KV head、词表 6400、image token id 12、image token 数 64。
- 对照 `trainer_utils.py` 确认：Pretrain 默认 `freeze_llm=2`，SFT 默认 `freeze_llm=1`，后者只重新开放第 0 和最后一个 Transformer block，加上 Projector。
- 复算 Projector 参数为 1,182,720；复算 93-token fp16 单样本 KV cache 约 1.09 MiB，64 个视觉 token 约贡献 0.75 MiB。
- 专业术语检查通过：Tensor、Token、Embedding、Linear、Softmax、Cross Entropy、Vision Transformer、Projector、Self-Attention、Causal Mask、GQA、RoPE、KV Cache、Prefill、Decode、LoRA、Quantization 和 Speculative Decoding 均在正文给出中文解释。
- 公式块 20 组，Markdown 代码围栏成对闭合，未发现 TODO/TBD 类未完成内容。
- 保留诚实边界：导师问题为基于公开方向的预测；不声称完整训练、标准 benchmark 或自行发明架构。

### Git 发布

- 讲义提交：`7a079d1 docs: add beginner-friendly MiniMind-V interview guide`。
- 推送目标：个人仓库 `https://github.com/cupkk/MiniMind-V.git` 的 `main` 分支。
- 推送结果：`93282ba..7a079d1 main -> main`。
- 首轮远端核验：本地和 `origin/main` 均为 `7a079d166cfeeea66f3013887d92f26b38553f1d`，工作区干净。
- 下一步：提交本条发布日志并再次核验最终远端状态，然后开始逐题模拟面试。
