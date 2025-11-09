import os, subprocess, sys
CFG_DIR = os.path.join(os.path.dirname(__file__), "configs")

def run(cfg):
    print(f"Running {cfg} ...")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "run_experiment.py"), "--config", os.path.join(CFG_DIR, cfg)]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run("baseline.yaml")
    run("no_pos.yaml")
    run("single_head.yaml")
    run("post_ln.yaml")
    run("no_dropout.yaml")