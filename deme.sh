seed=1
shot=10
NET=r101
setting="gfsod"

results_json='sadn_${setting}_${NET}_novel/tfa-like-DC/${shot}shot_seed${seed}/inference/coco_instances_results.json'

python3 tools/visualize_results.py   \
        --input $results_json \
        --out ./output/coco14_sadn_${setting}_${NET}_novel_${shot}shot_seed${seed}_vis_res  \
        --dataset  coco14_test_all
