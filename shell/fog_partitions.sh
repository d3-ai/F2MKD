#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_fogs=10
num_clients=2000
# dataset configuration
dataset="FashionMNIST"
fog_partitions=$1# "noniid-label#", "noniid-label#", "iid" 
# others
seed=1234

save_dir="./data/${dataset}/partitions/F_${num_fogs}_${fog_partitions}"

if [ ! -e "${save_dir}" ]; then
    mkdir -p "${save_dir}/logs/"
fi

python ./local/fog_partitions.py \
--num_fogs ${num_fogs} \
--partitions ${fog_partitions} \
--dataset ${dataset} \
--save_dir ${save_dir} \
--seed ${seed} \
1> "${save_dir}/logs/standard.log" \
2> "${save_dir}/logs/flower.log"