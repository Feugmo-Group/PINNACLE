# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

# python ../yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m5 --array=1
# python ../yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m5 --array=1
# python ../yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh --qos=m3 --array=1
# python ../yaml_to_sh.py sweeps/ENGD_woodbury.yaml sweeps/ENGD_woodbury.sh --qos=m5 --array=1
python ../yaml_to_sh.py sweeps/SPRING.yaml sweeps/SPRING.sh --qos=m5 --array=1

