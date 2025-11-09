import math, torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch.nn as nn
import torch.nn.functional as F
from .model import TransformerLM
from .utils import set_seed

@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 4
    block_size: int = 256
    dropout: float = 0.1
    pre_ln: bool = True
    use_pos: bool = True
    batch_size: int = 64
    max_epochs: int = 20
    steps_per_epoch: int = 200
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "outputs"

def get_cosine_scheduler(optimizer, base_lr, warmup, total_steps, min_lr=1e-5):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate(model: nn.Module, ds, steps=200, batch_size=64, device="cpu"):
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = ds.get_batch(batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses)/len(losses)

def build_model(vocab_size: int, cfg: Config) -> nn.Module:
    return TransformerLM(
        vocab_size=vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_ff=cfg.d_ff, n_layers=cfg.n_layers, block_size=cfg.block_size,
        drop=cfg.dropout, pre_ln=cfg.pre_ln, use_pos=cfg.use_pos
    )

def train_one_setting(cfg: Config, train_ds, val_ds, vocab_size: int):
    set_seed(cfg.seed)
    model = build_model(vocab_size, cfg).to(cfg.device)
    total_steps = cfg.max_epochs * cfg.steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = get_cosine_scheduler(opt, base_lr=cfg.lr, warmup=cfg.warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.device.startswith("cuda"))

    best_val = float("inf")
    hist = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        running = 0.0
        for _ in range(cfg.steps_per_epoch):
            x, y = train_ds.get_batch(cfg.batch_size, cfg.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.device.startswith("cuda")):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
        train_loss = running / cfg.steps_per_epoch
        val_loss = evaluate(model, val_ds, steps=100, batch_size=cfg.batch_size, device=cfg.device)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{cfg.save_dir}/checkpoints/best.pt")
        print(f"Epoch {epoch:02d}: train {train_loss:.3f} | val {val_loss:.3f} | ppl {math.exp(val_loss):.2f}")
    return best_val, hist, model