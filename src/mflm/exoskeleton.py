"""The Exoskeleton: MFLM Scout + Qwen Brain.

This module implements a hybrid architecture:
1. MFLM (Scout) reads the sequence and computes a field over tokens.
2. Adapter projects the MFLM field to Qwen's hidden dimension.
3. Qwen (Brain) receives its standard embeddings PLUS the adapter's output.
4. Qwen's core layers are frozen, but its attention matrices have LoRA 
   adapters attached, which learn to trust the MFLM field.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from mflm.model import MFLMConfig, MFLM


class ExoskeletonAdapter(nn.Module):
    """Bridges the gap between MFLM and Qwen."""
    def __init__(self, mflm_d: int, qwen_d: int):
        super().__init__()
        # Linear projection
        self.proj = nn.Linear(mflm_d, qwen_d, bias=False)
        # Initialize to zero so the hybrid starts identical to vanilla Qwen
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MFLMExoskeleton(nn.Module):
    def __init__(
        self, 
        mflm_cfg: MFLMConfig, 
        qwen_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lora_r: int = 8,
        lora_alpha: int = 32
    ):
        super().__init__()
        
        # 1. Scout: MFLM (initialized from scratch)
        self.mflm = MFLM(mflm_cfg)
        
        # 2. Brain: Qwen
        print(f"Loading Qwen model from {qwen_path} (this might take a minute)...")
        # Load in fp16 to save memory
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path, 
            torch_dtype=torch.float16,
            device_map=None # We will move it to device externally
        )
        
        # Freeze Qwen base
        for param in self.qwen.parameters():
            param.requires_grad = False
            
        # 3. Inject LoRA into Qwen's attention layers
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]  # Intercept attention queries/values
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        
        # 4. Bridge: Adapter
        qwen_d = self.qwen.config.hidden_size
        self.adapter = ExoskeletonAdapter(mflm_cfg.d_model, qwen_d)
        
        self.vocab_size = self.qwen.config.vocab_size

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the Exoskeleton.
        
        Args:
            input_ids: (B, T)
            labels: (B, T) optional
            
        Returns:
            loss (if labels provided) or logits (B, T, V)
        """
        # 1. Scout computation: get MFLM field states
        # return_hiddens=True returns the field states before the MFLM LM head
        # We don't need MFLM to predict tokens directly, just to build the field
        # The MFLM token embeddings might have different vocab size, but we pass input_ids.
        # Wait, MFLM's vocab_size MUST match Qwen's vocab_size (151936 for Qwen1.5)
        # We ensure this in the config.
        mflm_states = self.mflm(input_ids, return_hiddens=True)  # (B, T, mflm_d)
        
        # 2. Bridge
        bridge_signal = self.adapter(mflm_states)  # (B, T, qwen_d)
        
        # 3. Brain base embeddings
        # Qwen1.5 uses standard embeddings at qwen.model.embed_tokens
        qwen_base = self.qwen.get_base_model()
        qwen_embeddings = qwen_base.model.embed_tokens(input_ids)  # (B, T, qwen_d)
        
        # 4. Fusion!
        # The scout's signal is added to the brain's base understanding of the tokens.
        # Since adapter is initialized to 0, initially fused_states == qwen_embeddings.
        fused_states = qwen_embeddings + bridge_signal
        
        # 5. Brain computation
        # We pass the fused states directly into Qwen's transformer layers
        # input_ids is None because we provide inputs_embeds instead.
        outputs = self.qwen(
            inputs_embeds=fused_states,
            labels=labels
        )
        
        if labels is not None:
            return outputs.loss
        return outputs.logits
