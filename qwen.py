from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    model_dim: int = 2560
    head_dim: int = 128
    num_heads: int = 32
    num_kv_heads: int = 8
    num_layers: int = 36
    rope_base: float = 1000000.0
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    intermediate_size: int = 9728
    initialize_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    context_length: int = 40960
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        return self.weight * x.to(dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        exponent = torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim
        self.register_buffer(
            "inv_freq", 1.0 / (cfg.rope_base**exponent), persistent=False
        )

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        positions = position_ids[:, None, :].float()

        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)

    return torch.cat((-right, left), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg

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
        self.q_norm = Qwen3RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, _ = x.shape

        q = self.q_norm(
            self.q_proj(x).view(b, t, self.cfg.num_heads, self.cfg.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(x).view(b, t, self.cfg.num_kv_heads, self.cfg.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(x)
            .view(b, t, self.cfg.num_kv_heads, self.cfg.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )

        out = out.transpose(1, 2).reshape(b, t, -1)

        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.gate_proj = nn.Linear(
            cfg.model_dim, cfg.intermediate_size, bias=cfg.mlp_bias
        )
        self.up_proj = nn.Linear(
            cfg.model_dim, cfg.intermediate_size, bias=cfg.mlp_bias
        )
        self.down_proj = nn.Linear(
            cfg.intermediate_size, cfg.model_dim, bias=cfg.mlp_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.self_attn = Qwen3Attention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = Qwen3RMSNorm(cfg.model_dim, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.model_dim, cfg.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(cfg) for _ in range(cfg.num_layers)
        )
        self.norm = Qwen3RMSNorm(cfg.model_dim, cfg.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(cfg)

    def forward_embeddings(
        self,
        input_embeds: torch.Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        b, t, _ = input_embeds.shape

        if lengths is not None:
            assert lengths.shape == (b,)

        position_ids = torch.arange(0, t, dtype=torch.long).to(input_embeds.device)
        cos, sin = self.rotary_emb(input_embeds, position_ids.unsqueeze(0))

        x = input_embeds
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)

        return self.norm(x)

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        input_embeds: Tensor = self.embed_tokens(input_ids)
        return self.forward_embeddings(input_embeds, lengths)

    @staticmethod
    def from_pretrained(model_id: str, cfg: ModelConfig) -> "Qwen3Model":
        state = fetch_hf_state(model_id, dtype=cfg.dtype)
        state = {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.")
        }

        model = Qwen3Model(cfg).to(cfg.dtype)
        model.load_state_dict(state, strict=True)

        return model


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.model = Qwen3Model(cfg)
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
        input_ids: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        hidden = self.model(input_ids, lengths)

        return self.lm_head(hidden)


def fetch_hf_state(
    model_id: str, revision: str = "main", dtype: torch.dtype = torch.float32
) -> dict[str, Tensor]:
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(
        snapshot_download(model_id, revision=revision, allow_patterns=["*.safetensors"])
    )

    state = {}
    for shard in sorted(path.glob("*.safetensors")):
        state.update({k: v.to(dtype) for k, v in load_file(shard).items()})

    return state


def load_hf_weights(
    model: Qwen3ForCausalLM, model_id: str, revision: str = "main"
) -> Qwen3ForCausalLM:
    state = fetch_hf_state(model_id, revision)

    if model.cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    model.load_state_dict(state, strict=True)

    return model


def from_pretrained(
    model_id: str = "Qwen/Qwen3-4B",
    cfg: ModelConfig | None = None,
    dtype: torch.dtype = torch.float32,
) -> Qwen3ForCausalLM:
    model = Qwen3ForCausalLM(cfg or ModelConfig())
    load_hf_weights(model, model_id)

    return model.to(dtype).eval()
