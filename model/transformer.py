"""GPT-подобный декодер с нуля: RMSNorm, RoPE, causal self-attention, SwiGLU, weight tying, KV-кэш.

Из PyTorch берём nn.Linear, nn.Embedding, softmax и autograd; внимание, нормализация, позиционное
кодирование и MLP написаны здесь. Для обучения есть быстрый путь attention="sdpa" — flash-ядро
F.scaled_dot_product_attention: на контексте 1024 оно не хранит матрицу внимания T×T и экономит
память. Тесты сверяют его с ручным вниманием.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    context: int = 1024
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    mlp_hidden: int = 0          # 0 → ≈ 8/3·d_model, кратно 64: столько же параметров, сколько у MLP 4·d с GELU
    rope_base: float = 10000.0
    dropout: float = 0.0
    attention: str = "manual"    # "manual" — своё внимание, "sdpa" — flash-ядро PyTorch для обучения

    def __post_init__(self):
        if self.d_model % self.n_head or (self.d_model // self.n_head) % 2:
            raise ValueError("d_model должен делиться на n_head, размер головы — чётный (RoPE вращает пары)")
        if self.attention not in ("manual", "sdpa"):
            raise ValueError(f"attention={self.attention!r}: нужно 'manual' или 'sdpa'")
        if not self.mlp_hidden:
            self.mlp_hidden = 64 * math.ceil(8 * self.d_model / 3 / 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, d_model=64, context=128),   # для тестов
    "small": dict(n_layer=6, n_head=6, d_model=384),              # ~14M — запасной вариант, если корпуса мало
    "base": dict(n_layer=8, n_head=8, d_model=512),               # ~30M — основной
}


class RMSNorm(nn.Module):
    """x / rms(x) · g: как LayerNorm, но без вычитания среднего и без сдвига."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(context: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos и sin углов pos·θ_i, θ_i = base^(-2i/head_dim): форма (context, head_dim/2)."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(context, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Поворачивает пары (x_i, x_{i+d/2}) на угол pos·θ_i. Тогда q·k зависит только от расстояния
    между позициями, а не от самих позиций. x: (B, H, T, head_dim), cos/sin: (T, head_dim/2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.type_as(x)


class KVCache:
    """Ключи и значения уже обработанных позиций по слоям: при генерации шаг считает только новый токен."""

    def __init__(self, n_layer: int):
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    def length(self, layer: int = 0) -> int:
        """Сколько позиций уже в кэше слоя: внутри прохода нижние слои успевают дописать новые."""
        k = self.k[layer]
        return 0 if k is None else k.shape[2]

    def extend(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k[layer] is not None:
            k = torch.cat((self.k[layer], k), dim=2)
            v = torch.cat((self.v[layer], v), dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        H, hd = self.cfg.n_head, self.cfg.head_dim
        q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in self.qkv(x).split(C, dim=2))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            past = cache.length(self.layer)
            k, v = cache.extend(self.layer, k, v)
        # запрос i (абсолютная позиция past+i) видит ключи 0..past+i
        S = k.shape[2]
        if self.cfg.attention == "sdpa":
            p = self.cfg.dropout if self.training else 0.0
            if past == 0:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            else:  # is_causal в PyTorch выравнивает маску по левому краю — с кэшем нужна явная
                mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=past)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        else:
            y = self.drop(self.attention_weights(q, k, past)) @ v
        return self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))

    @staticmethod
    def attention_weights(q: torch.Tensor, k: torch.Tensor, past: int = 0) -> torch.Tensor:
        """softmax(q·kᵀ / √d) с причинной маской: (B, H, T, S). Отдельно — чтобы потом смотреть, куда смотрит модель."""
        T, S = q.shape[2], k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
        mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=past)
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores.float(), dim=-1).type_as(q)


class SwiGLU(nn.Module):
    """down(silu(gate(x)) · up(x)), silu(g) = g·σ(g). gate и up — одна матрица, так быстрее."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(g * torch.sigmoid(g) * u))


class Block(nn.Module):
    """Pre-norm: x + attn(norm(x)), затем x + mlp(norm(x)) — остаточный поток не нормируется."""

    def __init__(self, cfg: GPTConfig, layer: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: одна матрица на вход и на выход
        cos, sin = rope_tables(cfg.context, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # выходы attn и MLP складываются в остаточный поток 2·n_layer раз — уменьшаем их,
        # чтобы дисперсия потока не росла с глубиной (GPT-2)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # общая матрица считается один раз

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: KVCache | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx (B, T) → логиты (B, T, vocab). targets — следующие токены; -100 не входит в loss
        (так на SFT считаем ошибку только на ответе)."""
        B, T = idx.shape
        start = cache.length() if cache is not None else 0
        if start + T > self.cfg.context:
            raise ValueError(f"{start + T} токенов не влезают в контекст {self.cfg.context}")
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, eos_id: int | None = None, use_cache: bool = True,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Дописывает к idx (B, T) до max_new_tokens токенов. temperature=0 — жадный выбор."""
        cache = KVCache(self.cfg.n_layer) if use_cache else None
        out, step_input = idx, idx
        for _ in range(max_new_tokens):
            if out.shape[1] >= self.cfg.context:
                break
            logits, _ = self(step_input, cache=cache) if use_cache else self(out)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            out = torch.cat((out, nxt), dim=1)
            step_input = nxt
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return out

    # ---------- сохранение ----------

    def save(self, path: Path, **extra) -> None:
        """extra — что угодно из тензоров и примитивов: шаг, состояние оптимизатора, метрики."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "model": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu") -> tuple[GPT, dict]:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(GPTConfig(**ckpt.pop("config"))).to(device)
        model.load_state_dict(ckpt.pop("model"))
        return model, ckpt


def make_optimizer(model: nn.Module, lr: float, weight_decay: float,
                   betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    """AdamW; weight decay только на матрицах — нормы не тянем к нулю."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    fused = all(p.is_cuda for p in params)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
