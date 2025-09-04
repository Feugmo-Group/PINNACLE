# Launch all sweeps using the sbatch command
cd sweeps/

# Benchamrks
# sbatch SGD.sh
# sbatch Adam.sh
# sbatch ENGD.sh
# sbatch HessianFree.sh

# Woodbury ENGD
sbatch ENGD_woodbury.sh
sbatch SPRING.sh
# sbatch ENGD_woodbury_nystrom.sh
# sbatch SPRING_nystrom.sh
