# Launch all sweeps using the sbatch command
cd sweeps/

# Woodbury ENGD
# sbatch ENGD_woodbury_1k.sh
sbatch SPRING_1k.sh
# sbatch ENGD_woodbury_5k.sh
sbatch SPRING_5k.sh
# sbatch ENGD_woodbury_10k.sh
sbatch SPRING_10k.sh

sbatch ENGD_nystrom_1k.sh
sbatch SPRING_nystrom_1k.sh
sbatch ENGD_nystrom_5k.sh
sbatch SPRING_nystrom_5k.sh
sbatch ENGD_nystrom_10k.sh
sbatch SPRING_nystrom_10k.sh

