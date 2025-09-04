# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

# BENCHMARKS
# python ../yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m2 --array=50
# python ../yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=normal --array=50
# python ../yaml_to_sh.py sweeps/ENGD.yaml sweeps/ENGD.sh --qos=m --array=50
# python ../yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh --qos=m2 --array=50

# WOODBURY ENGD
# python ../yaml_to_sh.py sweeps/ENGD_woodbury.yaml sweeps/ENGD_woodbury.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING.yaml sweeps/SPRING.sh --qos=m4 --array=50
python ../yaml_to_sh.py sweeps/ENGD_nystrom.yaml sweeps/ENGD_nystrom.sh --qos=m --array=50
python ../yaml_to_sh.py sweeps/SPRING_nystrom.yaml sweeps/SPRING_nystrom.sh --qos=m3 --array=50
python ../yaml_to_sh.py sweeps/ENGD_pcg.yaml sweeps/ENGD_pcg.sh --qos=m3 --array=50
python ../yaml_to_sh.py sweeps/SPRING_pcg.yaml sweeps/SPRING_pcg.sh --qos=m3 --array=50
