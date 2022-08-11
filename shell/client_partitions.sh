#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_fogs=10
num_clients=100 #
# dataset configuration
dataset="FashionMNIST"
fog_partitions=$1 # "noniid-label2,1", "iid"
client_partitions=$2 # "noniid-label2,1", "iid" 
# others
seed=1234

data_dir="./data/${dataset}/partitions/F_${num_fogs}_${fog_partitions}/"
save_dir="./data/${dataset}/partitions/F_${num_fogs}_${fog_partitions}/C_${num_clients}_${client_partitions}"

if [ ! -e "${save_dir}" ]; then
    mkdir -p "${save_dir}/logs/"
fi

python ./local/client_partitions.py \
--num_fogs ${num_fogs} \
--num_clients ${num_clients} \
--partitions ${client_partitions} \
--dataset ${dataset} \
--data_dir ${data_dir} \
--save_dir ${save_dir} \
--seed ${seed} \
1> "${save_dir}/logs/standard.log" \
2> "${save_dir}/logs/flower.log"