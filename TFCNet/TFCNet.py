import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from peft import get_peft_config, get_peft_model, TaskType, LoraConfig, AdaLoraConfig
from einops import rearrange
from collections import OrderedDict
from layers.Embed import PatchEmbedding_temp
from layers.RevIN import RevIN
from layers.einops_modules import RearrangeModule
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from layers.TPECore import TPECore
from data_provider.data_factory import data_provider
import os
# Set DASHSCOPE_API_KEY in the environment before using Qwen.


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2308.08469.pdf
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # 根据参数决定是否应用物理约束
        self.apply_physical_constraints = getattr(args, 'apply_physical_constraints', False)
        print(f"📊 物理约束状态: {'启用' if self.apply_physical_constraints else '禁用'}")
        
        self.C = args.enc_in if args.return_single_feature == False else 1

        
 # 是否启用完整TPE（用于消融实验）
        self.use_full_tpe = getattr(args, 'use_full_tpe', True)
        
        if self.use_full_tpe:
            # 使用新的协同模块
            self.tpe_core = TPECore(
                d_model=args.d_model,
                dropout=args.dropout,
                num_patterns=getattr(args, 'num_patterns', 12),
                pattern_length=getattr(args, 'pattern_length', 24)
            )
            print("✅ 启用TPE核心模块（频率增强 + 时序记忆 + 协同注意力）")
            
            # 为了兼容性，保留这些变量但设为None
            self.memory_enable = False
            self.freq_enable = False
            self.temporal_memory = None
            self.freq_enhancer = None
        else:
            # 保留原来的独立模块（用于消融对比）
            self.memory_enable = getattr(args, 'memory_enable', True)
            if self.memory_enable:
                print("✅ 启用时间模式记忆模块")
                self.temporal_memory = TemporalPatternMemory(
                    d_model=args.d_model,
                    num_patterns=getattr(args, 'num_patterns', 10),
                    pattern_length=getattr(args, 'pattern_length', 24),
                    temperature=getattr(args, 'memory_temperature', 0.1)
                )
            else:
                print("🚫 禁用时间模式记忆模块")
                self.temporal_memory = None
            
            self.freq_enable = getattr(args, 'freq_enable', True)
            if self.freq_enable:
                print("✅ 启用频率增强模块")
                self.freq_enhancer = FrequencyEnhancer(
                    d_model=args.d_model,
                    dropout=args.dropout,
                    freq_ratio=0.5
                )
            else:
                print("🚫 禁用频率增强模块")
                self.freq_enhancer = None
            
            self.tpe_core = None
        
        # RevIN
        self.revin = RevIN(self.C, affine=True)

         # Input layer
        self._set_input_layer()

        # Output layer
        self._set_output_layer()

        # Model
        self._set_model()

        # Prompt text is prepared offline; model setup never calls a remote LLM.
        self.use_prompt = bool(getattr(args, "prompt_file", None))
        self.prompt_embeddings = None
        self.prompt_len = 0
        if self.use_prompt:
            self._load_prompt_embeddings()

    def _load_prompt_embeddings(self):
        """Load an offline summary and initialize trainable prefix embeddings."""
        import json

        path = Path(self.args.prompt_file)
        if path.suffix.lower() == ".json":
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = getattr(self.args, "data_name", None)
            if expected and record.get("dataset") != expected:
                raise ValueError("Prompt cache dataset does not match data_name")
            summary = record["summary"]
        else:
            summary = path.read_text(encoding="utf-8")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Prompt file must contain a non-empty summary")

        tokenizer = AutoTokenizer.from_pretrained(self.args.LLM_path)
        token_ids = tokenizer(summary.strip(), return_tensors="pt",
                              truncation=False).input_ids
        limit = getattr(self.args, "prompt_length", 0)
        if limit < 0:
            raise ValueError("prompt_length must be non-negative")
        if limit and token_ids.shape[1] > limit:
            raise ValueError("Summary exceeds prompt_length; no silent truncation")
        # Reserve room for the actual patch sequence as well as the prefix.
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context is None:
            context = getattr(self.model.config, "n_positions", None)
        if context and token_ids.shape[1] >= context:
            raise ValueError("Prompt leaves no room for historical patch tokens")
        embeddings = self.model.get_input_embeddings()
        with torch.no_grad():
            prefix = embeddings(token_ids.to(embeddings.weight.device)).detach().clone()
        self.prompt_embeddings = nn.Parameter(prefix, requires_grad=True)
        self.prompt_len = prefix.shape[1]

    def _set_input_layer(self):
        if self.args.LLM == "gpt2":
            pos_embed_type = "none"
        elif self.args.LLM == "llama":
            pos_embed_type = "learned"
        else:
            raise ValueError(f"Unsupported LLM: {self.args.LLM}")
        
        token_embed_type = (
            "linear"
            if getattr(self.args, "token_embed_type", None) is None
            else self.args.token_embed_type
        )
        temporal_embed_type = (
            "learned"
            if getattr(self.args, "temporal_embed_type", None) is None
            else self.args.temporal_embed_type
        )
        token_embed_kernel_size = (
            3
            if getattr(self.args, "token_embed_kernel_size", None) is None
            else self.args.token_embed_kernel_size
        )
        
        self.input_layer = PatchEmbedding_temp(
            self.args.C_t,
            self.args.d_model,
            self.args.patch_len,
            self.args.stride,
            self.args.dropout,
            pos_embed_type=pos_embed_type,
            token_embed_type=token_embed_type,
            kernel_size=token_embed_kernel_size,
            temporal_embed_type=temporal_embed_type,
            freq=self.args.freq,
        )

    def _set_output_layer(self):
        if self.args.task_name == "supervised_finetuning":
            self.output_layer = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "linear",
                            nn.Linear(
                                self.args.d_model, self.args.patch_len, bias=False
                            ),
                        ),
                        ("dropout", nn.Dropout(self.args.dropout)),
                    ]
                )
            )
        else:
            # 直接使用调试得到的实际 T_p 值
            T_p = 27
            
            self.output_layer = nn.Sequential(
                OrderedDict(
                    [
                        ("flatten", nn.Flatten(start_dim=1)),
                        (
                            "linear",
                            nn.Linear(self.args.d_model * T_p, self.args.pred_len, bias=False),
                        ),
                        ("dropout", nn.Dropout(self.args.dropout)),
                        (
                            "rearrange",
                            RearrangeModule("(B C) T_out -> B T_out C", C=self.C),
                        ),
                    ]
                )
            )

    def _set_model(self):
        # Load LLM
        if self.args.no_pretrain:
            config = AutoConfig.from_pretrained(self.args.LLM_path / "config.json")
            self.model = AutoModel.from_config(config)
            assert (
                self.args.no_freeze
            ), "If no_pretrain is True, no_freeze must be True."
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.LLM_path)

        # Only choose the first K layers
        if self.args.LLM == "gpt2":
            self.model.transformer.h = self.model.transformer.h[:self.args.first_k_layers]
        elif self.args.LLM == "llama":
        # LlamaModel has a different structure
            self.model.layers = self.model.layers[:self.args.first_k_layers]

        # Apply PEFT
        peft_method = self.args.peft_method
        if peft_method == "none":
            print("No PEFT applied.")

            if not self.args.no_freeze:  # we should freeze
                for name, param in self.model.named_parameters():
                    if any(term in name for term in ["wpe"]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        else:
            print(f"Apply PEFT: {peft_method}")

            if peft_method == "adalora" and self.args.peft_params_lora_dropout == 0:
            # adalora with dropout=0 is problematic
                self.args.peft_params_lora_dropout = 0.01
            peft_params = {
                "r": self.args.peft_params_r,
                "lora_alpha": self.args.peft_params_lora_alpha,
                "lora_dropout": self.args.peft_params_lora_dropout,
            }
            peft_params["task_type"] = TaskType.FEATURE_EXTRACTION
            peft_params["target_modules"] = ["c_attn"]
            peft_params["fan_in_fan_out"] = True

            if peft_method == "lora":
                peft_config = LoraConfig(**peft_params)
            elif peft_method == "adalora":
                peft_config = AdaLoraConfig(**peft_params)
            else:
                raise NotImplementedError

            self.model = get_peft_model(self.model, peft_config)

            if not self.args.no_freeze:
                for name, param in self.model.named_parameters():
                    if any(term in name for term in ["ln", "wpe"]):
                        param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = True

        # Linear probing
        if "lp" in self.args.ft_mode:
            print("Apply linear probing.")
            for param in self.revin.parameters():
                param.requires_grad = False
            for param in self.input_layer.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = False

    def linear_probe_to_fine_tuning(self):
        print("Linear probing to fine-tuning.")
        for param in self.revin.parameters():
            param.requires_grad = True
        for param in self.input_layer.parameters():
            param.requires_grad = True
        for name, param in self.model.named_parameters():
            if any(term in name for term in ["lora", "ia3"]):
                param.requires_grad = True

    
    def supervised_finetuning(self, x, x_mark):
        x = self.input_layer(x, x_mark)
        T_p = x.shape[1]
        
        # 【重构】使用TPE核心模块或原来的独立模块
        if self.use_full_tpe and self.tpe_core is not None:
            # 使用新的协同模块
            x, _ = self.tpe_core(x)
        else:
            # 使用原来的独立模块（用于消融对比）
            if self.freq_enable and self.freq_enhancer is not None:
                x = self.freq_enhancer(x)
            if self.memory_enable and self.temporal_memory is not None:
                x, _ = self.temporal_memory(x)
        
        x = self._prepend_prompt(x)
        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        y = outputs.hidden_states[-1]
        y = y[:, -T_p:, :]
        y = self.output_layer(y)
        return y

    def forecast(self, x, x_mark):
        x = self.revin(x, "norm")
        x = self.input_layer(x, x_mark)

        # 【重构】使用TPE核心模块或原来的独立模块
        if self.use_full_tpe and self.tpe_core is not None:
            # 使用新的协同模块
            x, _ = self.tpe_core(x)
        else:
            # 使用原来的独立模块（用于消融对比）
            if self.freq_enable and self.freq_enhancer is not None:
                x = self.freq_enhancer(x)
            if self.memory_enable and self.temporal_memory is not None:
                x, _ = self.temporal_memory(x)


        # 提示增强
        x = self._prepend_prompt(x)

        outputs = self.model(inputs_embeds=x, output_hidden_states=True)
        y = outputs.hidden_states[-1] 

            # ========== 添加调试代码 ==========
        print(f"🔍 y.shape before output_layer: {y.shape}")
        print(f"🔍 y.shape[1] (T_p actual) = {y.shape[1]}")
        print(f"🔍 d_model = {self.args.d_model}")
        print(f"🔍 Expected linear input features = {y.shape[1] * self.args.d_model}")
        print(f"🔍 Current linear weight shape = {self.output_layer[1].weight.shape}")
        # =================================
        y = self.output_layer(y)
        y = self.revin(y, "denorm")
        y = self._apply_output_constraints(y)
        return y

    def _apply_output_constraints(self, y):
        """统一应用物理约束"""
        if not self.apply_physical_constraints:
            return y  # 如果禁用物理约束，直接返回
        try:
            data_name = getattr(self.args, 'data_name', '').lower()
            
            if 'ett' in data_name:
                # ETT数据集的详细约束
                y = torch.clamp(y, -25.0, 80.0)  # 更合理的温度范围
                # 如果是多变量预测，对其他变量也应用约束
                if y.size(-1) > 1:
                    y[..., 1:] = torch.clamp(y[..., 1:], 0.0, 400.0)  # 功率非负
            elif 'weather' in data_name:
                y = torch.clamp(y, -50.0, 50.0)
            elif 'ecl' in data_name:
                y = torch.clamp(y, 0.0, 500.0)
            elif 'traffic' in data_name:
                y = torch.clamp(y, 0.0, 1.0)
            else:
                y = torch.clamp(y, -100.0, 100.0)
                
        except Exception as e:
            print(f"物理约束应用失败: {e}")
            y = torch.clamp(y, -100.0, 100.0)
    
        return y
    
    def _prepend_prompt(self, x):
        """将提示token拼接到序列前面"""
        if not self.use_prompt or not hasattr(self, 'prompt_embeddings') or self.prompt_embeddings is None:
            return x
        
        batch_size, seq_len, d_model = x.shape
        
        # 提示嵌入的形状是 (1, prompt_len, d_model)
        prompt_len = self.prompt_embeddings.shape[1]
        prompt = self.prompt_embeddings.expand(batch_size, prompt_len, -1)
        
        # 拼接：提示在前，原始序列在后
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context is None:
            context = getattr(self.model.config, "n_positions", None)
        if context and x.shape[1] + prompt_len > context:
            raise ValueError("Prompt plus patch tokens exceed backbone context length")
        enhanced_x = torch.cat([prompt, x], dim=1)
        
        return enhanced_x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.args.task_name == "supervised_finetuning":
            return self.supervised_finetuning(x_enc, x_mark_enc)
        elif self.args.task_name == "long_term_forecast":
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise NotImplementedError