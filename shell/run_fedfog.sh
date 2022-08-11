#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_rounds=10
fraction_fit=1
num_fogs=10
client_pool=100

# dataset configuration
dataset="FashionMNIST"
model_name="tinyCNN"
fog_partitions=$1 # "noniid", "iid"
client_partitions=$2

# Client configuration
lr=0.05
client_batch_size=60
client_epochs=5

# others
seed=1234
data_dir="./data/${dataset}/partitions/F_${num_fogs}_${fog_partitions}"
exp_dir="./exp/${dataset}/FedFog_${model_name}/F_${num_fogs}_${fog_partitions}_C_${client_pool}_${client_partitions}/F_${fraction_fit}_R_${num_rounds}_CB_${client_batch_size}_CE_${client_epochs}_lr_${lr}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./local/run_fedfog.py \
--num_rounds ${num_rounds} \
--fraction_fit ${fraction_fit} \
--num_fogs ${num_fogs} \
--client_pool ${client_pool} \
--dataset ${dataset} \
--client_partitions ${client_partitions} \
--model_name ${model_name} \
--client_epochs ${client_epochs} \
--client_batch_size ${client_batch_size} \
--lr ${lr} \
--save_dir ${exp_dir} \
--data_dir ${data_dir} \
--seed ${seed} \
1> "${exp_dir}/logs/standard.log" \
2> "${exp_dir}/logs/flower.log"