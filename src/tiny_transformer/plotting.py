import matplotlib.pyplot as plt
def plot_curves(histories, labels, fname):
    plt.figure(figsize=(7,5))
    for hist, lab in zip(histories, labels):
        plt.plot(hist["val_loss"], label=lab, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Val Loss (nats/token)")
    plt.title("Validation Loss")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=160); print(f"Saved {fname}")
def plot_train_val(hist, fname):
    plt.figure(figsize=(7,5))
    plt.plot(hist["train_loss"], label="Train", linewidth=2)
    plt.plot(hist["val_loss"], label="Val", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Val")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=160); print(f"Saved {fname}")