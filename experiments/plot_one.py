# experiments/plot_one.py  （纯绘图，无需导入 tiny_transformer/torch）
import os, json, argparse
import matplotlib.pyplot as plt

def plot_train_val(hist, out_path):
    plt.figure(figsize=(7,5))
    plt.plot(hist["train_loss"], label="Train", linewidth=2)
    plt.plot(hist["val_loss"], label="Val", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Val")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160); print(f"Saved {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="path to outputs\\exp-...")
    args = ap.parse_args()
    with open(os.path.join(args.exp, "history.json"), "r", encoding="utf-8") as f:
        j = json.load(f)
    hist = j.get("history", j)  # 兼容两种格式
    out_path = os.path.join(args.exp, "figures", "train_val.png")
    plot_train_val(hist, out_path)

if __name__ == "__main__":
    main()