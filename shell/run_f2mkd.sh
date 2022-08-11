#!/usr/bin/bash

. ./shell/path.sh

# FL system configuration
num_rounds=500
fraction_fit=1
num_fogs=10
client_pool=100

# dataset configuration
dataset="FashionMNIST"
model_name="tinyCNN"
fog_partitions=$1 # "noniid", "iid"
client_partitions=$2

# Fog configuration
fog_batch_size=60
fog_global_epochs=5
fog_client_epochs=4
alpha=$3
beta=$4

# Client configuration
lr=0.05
client_batch_size=60
client_epochs=1

# others
seed=1234
data_dir="./data/${dataset}/partitions/F_${num_fogs}_${fog_partitions}"
exp_dir="./exp/${dataset}/FedMKD_${model_name}/F_${num_fogs}_${fog_partitions}_C_${client_pool}_${client_partitions}/F_${fraction_fit}_R_${num_rounds}_FB_${fog_batch_size}_FGE_${fog_global_epochs}_FCE_${fog_client_epochs}_CB_${client_batch_size}_CE_${client_epochs}_lr_${lr}_alpha_${alpha}_beta_${beta}_S_${seed}"

if [ ! -e "${exp_dir}" ]; then
    mkdir -p "${exp_dir}/logs/"
    mkdir -p "${exp_dir}/models/"
    mkdir -p "${exp_dir}/metrics/"
fi

python ./local/run_fedmkd.py \
--num_rounds ${num_rounds} \
--fraction_fit ${fraction_fit} \
--num_fogs ${num_fogs} \
--client_pool ${client_pool} \
--dataset ${dataset} \
--client_partitions ${client_partitions} \
--model_name ${model_name} \
--client_epochs ${client_epochs} \
--client_batch_size ${client_batch_size} \
--fog_global_epochs ${fog_global_epochs} \
--fog_client_epochs ${fog_client_epochs} \
--fog_batch_size ${fog_batch_size} \
--lr ${lr} \
--alpha ${alpha} \
--beta ${beta} \
--save_dir ${exp_dir} \
--data_dir ${data_dir} \
--seed ${seed} \
1> "${exp_dir}/logs/standard.log" \
2> "${exp_dir}/logs/flower.log"