import subprocess
import sys

def run_sweep():
    cmd = [
        sys.executable, "main.py", 
        "--multirun",
        "training.weight_strat=ntk,hybrid_ntk_batch",
        "scheduler.type=RLROP,exponential,none"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_sweep()