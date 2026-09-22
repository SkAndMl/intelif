from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    model_dim: int = 960
    head_dim: int = 64
    num_heads: int = 15
    num_kv_heads: int = 5
    num_layers: int = 32
    rope_base: float = 100000.0
    vocab_size: int = 49152
    tie_word_embeddings: bool = True
    intermediate_size: int = 2560
    initialize_range: float = 0.02
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    context_length: int = 8192

    def __post_init__(self):
        assert self.model_dim == self.head_dim * self.num_heads
        assert self.num_heads % self.num_kv_heads == 0

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        return self.weight * x.to(dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        exponent = torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim
        self.register_buffer("inv_freq", 1.0 / (cfg.rope_base**exponent), persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()

        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)

    return torch.cat((-right, left), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: Tensor, num_groups: int) -> Tensor:
    b, h, t, d = x.shape

    return x[:, :, None].expand(b, h, num_groups, t, d).reshape(b, h * num_groups, t, d)


class LlamaAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.scaling = cfg.head_dim**-0.5

        self.q_proj = nn.Linear(
            in_features=cfg.model_dim,
            out_features=cfg.head_dim * cfg.num_heads,
            bias=cfg.attention_bias,
        )
        self.k_proj = nn.Linear(
            in_features=cfg.model_dim,
            out_features=cfg.head_dim * cfg.num_kv_heads,
            bias=cfg.attention_bias,
        )
        self.v_proj = nn.Linear(
            in_features=cfg.model_dim,
            out_features=cfg.head_dim * cfg.num_kv_heads,
            bias=cfg.attention_bias,
        )
        self.o_proj = nn.Linear(
            in_features=cfg.head_dim * cfg.num_heads,
            out_features=cfg.model_dim,
            bias=cfg.attention_bias,
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.cfg.num_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.cfg.num_kv_heads, self.cfg.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.cfg.num_kv_groups)
        v = repeat_kv(v, self.cfg.num_kv_groups)

        weights = (q @ k.transpose(2, 3)) * self.scaling

        if mask is not None:
            weights = weights + mask

        weights = weights.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        out = (weights @ v).transpose(1, 2).reshape(b, t, -1)

        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.gate_proj = nn.Linear(cfg.model_dim, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.model_dim, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.model_dim, bias=cfg.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.self_attn = LlamaAttention(cfg)
        self.mlp = LlamaMLP(cfg)
        self.input_layernorm = LlamaRMSNorm(cfg.model_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.model_dim, cfg.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


def build_causal_mask(attention_mask: Tensor | None, t: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    minimum = torch.finfo(dtype).min
    positions = torch.arange(t, device=device)
    mask = torch.where(positions[None, :] > positions[:, None], minimum, 0.0).to(dtype)
    mask = mask[None, None].expand(1 if attention_mask is None else attention_mask.shape[0], 1, t, t)

    if attention_mask is not None:
        padding = torch.where(attention_mask.bool(), 0.0, minimum).to(dtype)
        mask = mask + padding[:, None, None, :]

    return mask


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList(LlamaDecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = LlamaRMSNorm(cfg.model_dim, cfg.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(cfg)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        b, t, _ = inputs_embeds.shape

        if position_ids is None:
            position_ids = torch.arange(t, device=inputs_embeds.device).expand(b, t)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        mask = build_causal_mask(attention_mask, t, inputs_embeds.dtype, inputs_embeds.device)

        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initialize_range)

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        hidden = self.model(input_ids, attention_mask, position_ids, inputs_embeds)

        return self.lm_head(hidden)


def load_hf_weights(model: LlamaForCausalLM, model_id: str, revision: str = "main") -> LlamaForCausalLM:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    state = load_file(hf_hub_download(model_id, "model.safetensors", revision=revision))
    state = {k: v.to(torch.float32) for k, v in state.items()}

    if model.cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    model.load_state_dict(state, strict=True)

    return model


def from_pretrained(
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    cfg: ModelConfig | None = None,
    dtype: torch.dtype = torch.float32,
) -> LlamaForCausalLM:
    model = LlamaForCausalLM(cfg or ModelConfig())
    load_hf_weights(model, model_id)

    return model.to(dtype).eval()
