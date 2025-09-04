# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

python ../yaml_to_sh.py sweeps/ENGD_woodbury_1k.yaml sweeps/ENGD_woodbury_1k.sh --qos=m --array=20
python ../yaml_to_sh.py sweeps/SPRING_1k.yaml sweeps/SPRING_1k.sh --qos=m2 --array=20
python ../yaml_to_sh.py sweeps/ENGD_woodbury_5k.yaml sweeps/ENGD_woodbury_5k.sh --qos=m3 --array=20
python ../yaml_to_sh.py sweeps/SPRING_5k.yaml sweeps/SPRING_5k.sh --qos=m4 --array=20
python ../yaml_to_sh.py sweeps/ENGD_woodbury_10k.yaml sweeps/ENGD_woodbury_10k.sh --qos=m --array=20
python ../yaml_to_sh.py sweeps/SPRING_10k.yaml sweeps/SPRING_10k.sh --qos=m2 --array=20

python ../yaml_to_sh.py sweeps/ENGD_nystrom_1k.yaml sweeps/ENGD_nystrom_1k.sh --qos=m3 --array=20
python ../yaml_to_sh.py sweeps/SPRING_nystrom_1k.yaml sweeps/SPRING_nystrom_1k.sh --qos=m4 --array=20
python ../yaml_to_sh.py sweeps/ENGD_nystrom_5k.yaml sweeps/ENGD_nystrom_5k.sh --qos=m --array=20
python ../yaml_to_sh.py sweeps/SPRING_nystrom_5k.yaml sweeps/SPRING_nystrom_5k.sh --qos=m2 --array=20
python ../yaml_to_sh.py sweeps/ENGD_nystrom_10k.yaml sweeps/ENGD_nystrom_10k.sh --qos=m3 --array=20
python ../yaml_to_sh.py sweeps/SPRING_nystrom_10k.yaml sweeps/SPRING_nystrom_10k.sh --qos=m4 --array=20
