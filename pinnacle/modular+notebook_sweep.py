#!/usr/bin/env python3
"""
Simple script to run full experiment pipeline:
1. Run one modular experiment
2. Run notebook parameter sweep on ntk steps
3. Run single notebook experiment
"""
import subprocess
import sys
import os

def run_single_modular():
    """Run a single modular experiment"""
    print("🚀 Step 1: Running single modular experiment...")
    print("=" * 50)
    
    # Run your sweep.py (but it should be modified to run just once)
    result = subprocess.run([sys.executable, "sweep.py"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Single modular experiment completed successfully!")
    else:
        print("❌ Single modular experiment failed:")
        print(result.stderr)
    
    print(result.stdout)

def run_notebook_sweep():
    """Run notebook sweep on NTK steps"""
    print("\n🔬 Step 2: Running notebook NTK sweep...")
    print("=" * 50)
    
    # Install papermill if needed
    subprocess.run([sys.executable, "-m", "pip", "install", "papermill"],
                   capture_output=True, check=False)
    
    # NTK steps to sweep over
    ntk_steps_values = [2000, 5000, 7000]
    
    notebook_template = "NTK+ALPinnacle.ipynb"  # Your notebook name
    
    for i, ntk_steps in enumerate(ntk_steps_values):
        output_notebook = f"outputs/notebook_sweep_ntk_{ntk_steps}.ipynb"
        
        print(f"   Running NTK sweep {i+1}/{len(ntk_steps_values)}: ntk_steps={ntk_steps}")
        
        cmd = [
            "papermill",
            notebook_template,
            output_notebook,
            "-p", "ntk_steps", str(ntk_steps)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"   ✅ Completed ntk_steps={ntk_steps}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed ntk_steps={ntk_steps}")
            print(f"   Error: {e}")

def run_single_notebook():
    """Run a single notebook experiment with default parameters"""
    print("\n📓 Step 3: Running single notebook experiment...")
    print("=" * 50)
    
    notebook_template = "ALPinnacle.ipynb"
    output_notebook = "outputs/notebook_single_run.ipynb"
    
    print("   Running single notebook with default parameters...")
    
    cmd = [
        "papermill",
        notebook_template,
        output_notebook
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("   ✅ Single notebook run completed!")
    except subprocess.CalledProcessError as e:
        print("   ❌ Single notebook run failed:")
        print(f"   Error: {e}")

def main():
    """Run the complete pipeline"""
    print("🎯 Starting Modified PINNACLE Experiment Pipeline")
    print("=" * 60)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Step 2: Run notebook sweep
    run_notebook_sweep()
    
    # Step 3: Run single notebook
    run_single_notebook()
    
    print("\n🎉 Complete pipeline finished!")
    print("📁 Check 'outputs/' folder for all results")
    print("   📊 Modular result: outputs/experiments/...")
    print("   📓 Notebook sweep: outputs/notebook_sweep_ntk_*.ipynb")
    print("   📓 Single notebook: outputs/notebook_single_run.ipynb")

if __name__ == "__main__":
    main()