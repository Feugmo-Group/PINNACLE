# Launch all sweeps using the sbatch command
cd sweeps/

# sbatch SGD.sh
# sbatch Adam.sh
# sbatch HessianFree.sh
# sbatch ENGD_woodbury.sh
sbatch SPRING.sh