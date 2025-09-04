# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

# python ../yaml_to_sh.py sweeps/SPRING_100.yaml sweeps/SPRING_100.sh --qos=m2 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_500.yaml sweeps/SPRING_500.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_1k.yaml sweeps/SPRING_1k.sh --qos=m4 --array=50

# python ../yaml_to_sh.py sweeps/SPRING_nystrom_100.yaml sweeps/SPRING_nystrom_100.sh --qos=m2 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_nystrom_500.yaml sweeps/SPRING_nystrom_500.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_nystrom_1k.yaml sweeps/SPRING_nystrom_1k.sh --qos=m4 --array=50

python ../yaml_to_sh.py sweeps/SPRING_naive_1k.yaml sweeps/SPRING_naive_1k.sh --qos=m --array=50
# python ../yaml_to_sh.py sweeps/SPRING_naive_500.yaml sweeps/SPRING_naive_500.sh --qos=m2 --array=50
python ../yaml_to_sh.py sweeps/SPRING_naive_100.yaml sweeps/SPRING_naive_100.sh --qos=m3 --array=50