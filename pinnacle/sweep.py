import subprocess
import sys

def run_sweep():
    cmd = [
        sys.executable, "main.py", 
        "--multirun",
        "ntk_steps = "
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_sweep()