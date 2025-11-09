# experiments/plot_results.py  (不依赖 torch)
import os, json, argparse
import matplotlib.pyplot as plt

def load_hist(exp_dir):
    with open(os.path.join(exp_dir, "history.json"), "r", encoding="utf-8") as f:
        j = json.load(f)
    cfg = j.get("config", {})
    hist = j.get("history", j)
    if not cfg:  # 兜底
        cfg = {}
    if not hist:
        hist = {}
    # 生成友好标签
    lab = f"h={cfg.get('n_heads')}, pos={cfg.get('use_pos')}, preLN={cfg.get('pre_ln')}, drop={cfg.get('dropout')}"
    return lab, hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out",  default="outputs/figures/ablation_val_curves.png")
    args = ap.parse_args()
    exp_dirs = [os.path.join(args.root, d) for d in os.listdir(args.root) if d.startswith("exp-")]
    exp_dirs.sort()
    labels, histories = [], []
    for d in exp_dirs:
        try:
            lab, hist = load_hist(d)
            labels.append(lab); histories.append(hist)
        except FileNotFoundError:
            pass
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(8,5))
    for lab, hist in zip(labels, histories):
        plt.plot(hist["val_loss"], label=lab, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Val Loss"); plt.title("Ablation: Validation Curves")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(args.out, dpi=160); print("Saved", args.out)

if __name__ == "__main__":
    main()