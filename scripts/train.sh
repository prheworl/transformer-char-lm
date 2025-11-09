$ErrorActionPreference = "Stop"
# 可选：未 pip install -e . 的话，打开下面一行
# $env:PYTHONPATH = ".\src"
py -m experiments.run_experiment --config experiments\configs\baseline.yaml