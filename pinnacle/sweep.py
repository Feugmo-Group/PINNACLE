# sweep.py
import subprocess
import sys

def run_sweep():
    cmd = [
        sys.executable, "main.py", 
        "--multirun",
        "training.ntk_steps=1000,2000,3000,4000,5000,6000,7000,8000,9000,10000"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_sweep()