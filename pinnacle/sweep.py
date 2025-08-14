import subprocess
import sys

def run_sweep():
    cmd = [
        sys.executable, "pinnacle/main.py",
        "--multirun",
        "training.weight_strat=uniform,batch_size,ntk"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_sweep()