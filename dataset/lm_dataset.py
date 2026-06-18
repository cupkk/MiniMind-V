import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    # 这样做相当于轻量数据增强：同一类问答有时带 system prompt，有时不带，
    # 让模型更适应真实聊天场景中不同的对话开头。
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    # 有些数据会带空的 <think></think>，大多数时候删掉，少量保留用于兼容带思考格式的数据。
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        # Parquet 中每行包含 conversations 和 image_bytes。
        # 这里一次读成 Arrow Table，优点是随机索引快；缺点是大数据集会占用较多内存。
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        # 一个 <image> 会被展开成 64 个 <|image_pad|>。
        # 后续 model_vlm.count_vision_proj 会用 64 个视觉 token 替换这些 token 的 embedding。
        self.image_special_token = image_special_token * image_token_len
        # bos_id/eos_id 用来定位 assistant 回复段落。
        # 只有 assistant 说的话会作为 label 参与 loss，user/system prompt 不参与监督。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            # 数据里通常用 <image> 表示图像位置；模型实际需要 64 个 image pad token 占位。
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        # tokenizer 的 chat_template 会把 role/content 转成模型训练时看到的完整文本格式。
        # 这一步非常关键：训练和推理必须使用同一种模板，否则模型会学到/看到不同分布。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到 assistant 回复开始标记：<bos>assistant\n。
            # 从它后面开始，到 eos 结束，才是模型需要学习预测的答案。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # label 与 input_ids 等长；Causal LM 内部会 shift，所以这里把答案 token 原样放入 labels。
                # -100 的位置会被 cross_entropy 忽略，避免模型被训练去复述用户问题。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        # 支持单图和多图样本：单图是 bytes，多图可能是 bytes list。
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]
        
        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        # 截断到 max_length 后再 pad 到固定长度，方便 DataLoader 直接 stack 成 batch。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        # image_bytes 是 parquet 中保存的 JPEG/PNG 二进制；这里恢复为 PIL Image 后交给 SigLIP processor。
        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            # HuggingFace processor 通常返回字典，例如 {'pixel_values': tensor(...)}。
            # 多张图时按 key 拼起来，保持后续 collate 能继续按字典处理。
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data

# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        pf = pq.ParquetFile(path); n = pf.num_row_groups; t = pa.concat_tables([pf.read_row_group(i * n // 5).slice(0, 1) for i in range(5)]); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py(); img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{pf.metadata.num_rows}条')
