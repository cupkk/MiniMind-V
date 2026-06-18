import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        # image_special_token 是文本 prompt 中的图像占位符；训练前会把一个 <image>
        # 展开成 64 个 <|image_pad|>，和 SigLIP2 P32 输出的 64 个 patch token 一一对应。
        self.image_special_token = image_special_token
        # image_ids 是 tokenizer 中 <|image_pad|> 的 token id。默认 12 来自本仓库 tokenizer。
        # forward 时会扫描 input_ids，找到连续的 image_ids，并用视觉 embedding 替换这些位置。
        self.image_ids = image_ids
        # SigLIP2 vision encoder 输出维度。当前 siglip2-base-p32-256-ve 的 hidden size 是 768。
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        # 一张图转换成多少个视觉 token。256x256 图像 / 32x32 patch = 8x8 = 64。
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        # 这个 projector 是 VLM 的“翻译器”：把视觉 encoder 的特征空间映射到 LLM 的 hidden space。
        # MiniMind-V 采用 LLaVA-1.5 类似的两层 MLP，而不是只用一个 Linear。
        # 输入形状通常是 [batch, 64, image_hidden_size]，输出是 [batch, 64, llm_hidden_size]。
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)

# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMConfig()
        # 先初始化父类 MiniMindForCausalLM，得到纯文本 LLM 主干：token embedding、Transformer layers、lm_head。
        super().__init__(self.config)
        # vision_encoder 只负责提图像特征；processor 负责把 PIL image 变成模型需要的 pixel_values。
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数。训练时不改 SigLIP2，只训练 projector 和部分 LLM。
        # 这样成本低，也避免小数据把视觉 encoder 原有能力破坏掉。
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        # SigLIP processor 期望 RGB 图像；RGBA/LA 多了透明通道，先转成 RGB 保持输入一致。
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        # DataLoader collate 后可能产生 [batch, 1, C, H, W] 这类多一维的张量；
        # 这里 squeeze 掉长度为 1 的图片数量维，适配 HuggingFace SiglipVisionModel。
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        # vision_encoder 全程冻结，所以不需要梯度；节省显存和计算图开销。
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        # last_hidden_state 形状通常为 [batch, 64, 768]，对应 64 个 patch token。
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        # 关键步骤：把文本 embedding 中 <|image_pad|> 的位置替换成视觉 embedding。
        # tokens: [batch, seq_len]，h: [batch, seq_len, hidden]。
        # vision_tensors: 单图时 [batch, 64, hidden]；多图时 [batch, num_images, 64, hidden]。
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    # 找到一段连续的 image pad token，例如 64 个 <|image_pad|>。
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        # 用第 k 张图的视觉 token 替换这段占位符的 embedding。
                        # 如果占位符长度不是 64，也按实际占位长度切片，避免形状不匹配。
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            # 只在生成的第一步注入图像特征。后续自回归生成使用 KV cache，
            # 图像 token 已经在过去上下文里，不需要每生成一个 token 都重新编码图片。
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    # 多图 batch：processor 输出字典里的张量形状约为 [batch, num_images, C, H, W]。
                    # 先 flatten 成 [batch*num_images, C, H, W] 送入 vision_encoder，再 reshape 回来。
                    bs, num = sample_val.shape[:2]
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)).view(bs, num, self.config.image_token_len, -1)
                else:
                    # 单图 batch：得到 [batch, 64, hidden]。
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder))
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                # 非字典输入时逐张图编码，最后堆成 [batch, num_images, 64, hidden]。
                vision_tensors = torch.stack([self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Causal LM 标准训练目标：第 t 个位置的 logits 预测第 t+1 个 token。
            # labels 中的 -100 会被忽略，因此只有 assistant 回复部分参与 loss。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            # 一张图片生成多条候选回答时，文本 input 会 repeat，图像张量也必须同步 repeat。
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
