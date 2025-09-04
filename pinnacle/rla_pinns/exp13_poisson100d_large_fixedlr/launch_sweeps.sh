# Launch all sweeps using the sbatch command
cd sweeps/

# sbatch SPRING_1k.sh
# sbatch SPRING_500.sh
# sbatch SPRING_100.sh

# sbatch SPRING_nystrom_1k.sh
# sbatch SPRING_nystrom_500.sh
# sbatch SPRING_nystrom_100.sh

sbatch SPRING_naive_100.sh
sbatch SPRING_naive_500.sh
sbatch SPRING_naive_1k.sh

