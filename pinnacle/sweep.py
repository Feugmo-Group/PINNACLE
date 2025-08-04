import subprocess
import sys

def run_sweep():
    cmd = [
        sys.executable, "main.py",
        # Remove "--multirun" for single run
        "experiment.name=base_reduced_ntk_no_holes_no_sched_residual_tests"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_sweep()