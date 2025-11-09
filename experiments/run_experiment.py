import os, math, yaml, argparse, torch
from tiny_transformer import Config, train_one_setting, evaluate
from tiny_transformer.data import download_data, build_vocab, make_splits, CharDataset
from tiny_transformer.utils import ensure_dir, save_json, now_str

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    cfg = Config()
    for k,v in y.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to yaml config")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # prepare dirs
    stamp = now_str()
    cfg.save_dir = os.path.join(cfg.save_dir, f"exp-{stamp}")
    ensure_dir(cfg.save_dir); ensure_dir(os.path.join(cfg.save_dir, "checkpoints")); ensure_dir(os.path.join(cfg.save_dir, "figures"))

    # data
    text = download_data()
    stoi, itos, encode, decode = build_vocab(text)
    encoded = encode(text)
    train_raw, val_raw, test_raw = make_splits(encoded, split=(0.9,0.05,0.05))
    train_ds = CharDataset(train_raw, cfg.block_size)
    val_ds = CharDataset(val_raw, cfg.block_size)
    test_ds = CharDataset(test_raw, cfg.block_size)
    vocab_size = len(stoi)

    # train
    best_val, hist, model = train_one_setting(cfg, train_ds, val_ds, vocab_size)
    test_loss = evaluate(model, test_ds, steps=200, batch_size=cfg.batch_size, device=cfg.device)

    # save results
    save_json({"best_val": best_val, "test_loss": test_loss, "test_ppl": math.exp(test_loss)}, os.path.join(cfg.save_dir, "metrics.json"))
    save_json({"config": vars(cfg), "history": hist}, os.path.join(cfg.save_dir, "history.json"))
    print(f"Done. Test: loss={test_loss:.3f}, ppl={math.exp(test_loss):.2f}")

if __name__ == "__main__":
    main()