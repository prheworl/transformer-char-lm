import os, math, time, random, json
from dataclasses import asdict
import torch
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_str(): return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)