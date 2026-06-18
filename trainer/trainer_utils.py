"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM


def get_model_params(model, config, ignore_patterns=['vision_encoder']):
    # 统计参数量时默认忽略 vision_encoder，因为 SigLIP2 是外部冻结模块，
    # README 中的 65M 主要指 LLM + projector 这部分可控模型规模。
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    # 简单 cosine schedule，最低学习率约为初始 lr 的 10%。
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    
    # torchrun 会注入 RANK/LOCAL_RANK/WORLD_SIZE 等环境变量。
    # 检测到这些变量后，初始化 NCCL 进程组并把当前进程绑定到对应 GPU。
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_llm=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)
    
    if from_weight != 'none':
        # 原生 PyTorch 权重命名规则：
        # llm_768.pth / pretrain_vlm_768.pth / sft_vlm_768.pth / *_moe.pth。
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    
    # 1、全部冻结，只打开vision_proj梯度
    # 先把除 projector 外的参数全冻结，后面再按 freeze_llm 策略选择性解冻 LLM。
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    # 2、判断策略
    if freeze_llm == 0:
        # 全参训练：除 vision_encoder 外，LLM 和 projector 都可训练。成本最高，也最容易遗忘语言能力。
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        # SFT 默认策略：训练 projector + LLM 第 0 层和最后一层。
        # 第 0 层负责最早融合视觉 token，最后一层影响输出分布和回答风格。
        last_idx = vlm_config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        # Pretrain 默认策略：只训练 projector，让视觉特征先对齐到语言空间，不扰动 LLM。
        pass

    get_model_params(model, vlm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 移除vision_encoder参数（不需要保存，因为是预训练的）
        # 保存 VLM 权重时不保存 SigLIP2，复现时只要重新下载同一个 vision encoder 即可。
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        # 临时文件 + os.replace 是原子保存策略，避免训练中断时留下半截坏权重。
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            # resume 文件保存完整训练状态：模型、优化器、epoch、step、world_size。
            # 和 out/*.pth 不同，resume 文件是为了继续训练，不是为了最终推理发布。
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # 续训时 GPU 数量变了，每个 step 消耗的样本数也变了，所以按 world_size 比例换算 step。
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def vlm_collate_fn(batch):
    # Dataset 返回单条样本；collate_fn 把样本列表合并成 batch。
    # 文本直接 stack，图像既可能是 processor 字典，也可能是普通 tensor。
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 断点续训时跳过已经训练过的 batch，避免同一个 epoch 内重复训练。
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

