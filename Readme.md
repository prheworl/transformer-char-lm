# Transformer Char LM (Tiny Shakespeare)

从零搭建字符级 Transformer 语言模型，并在 Tiny Shakespeare 上进行消融实验。

## 安装
```bash
pip install -e .
pip install -r requirements.txt
```

# 运行baseline

```bash
python -m experiments.run_experiment --config experiments/configs/baseline.yaml
```

# 运行消融

```bash
set -e
python -m experiments.run_ablation
```
