import argparse
import json
import random
import numpy as np
import torch

from dataset_app.common import (
    create_noniid,
    load_mnist,
    load_fmnist,
    load_cifar10,
    create_iid,
    create_noniid_dir,
    write_json,
)

from pathlib import Path

parser = argparse.ArgumentParser("Create dataset partitions")

parser.add_argument("--num_fogs", type=int, required=True, help="The number of clients")
parser.add_argument("--num_clients", type=int, required=True, help="The number of clients per fog")
parser.add_argument("--partitions", type=str, required=True, help="dataset partitions")
parser.add_argument("--dataset", type=str, required=True, choices=["FashionMNIST","MNIST","CIFAR10"], help="dataset name")
parser.add_argument("--data_dir", type=str, required=True, help="data directory")
parser.add_argument("--save_dir", type=str, required=True, help="save directory")
parser.add_argument("--seed", type=int, required=True, help="random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    set_seed(args.seed)
    num_fogs = args.num_fogs
    num_parties = args.num_clients
    print(num_parties)
    dataset = args.dataset
    partitions = args.partitions
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    if dataset == "MNIST":
        _, y_train, _, y_test = load_mnist()
    elif dataset == "FashionMNIST":
        _, y_train, _, y_test = load_fmnist()
    elif dataset == "CIFAR10":
        _, y_train, _, y_test = load_cifar10()
    else:
        raise NotImplementedError(f"{dataset} is no implemented")

    train_json_path = data_dir / "train.json"
    with open(train_json_path, "r") as f:
        fog_train_json_data = json.load(f) 
    
    test_json_path = data_dir / "test.json"
    with open(test_json_path, "r") as f:
        fog_test_json_data = json.load(f) 

    for fid in range(num_fogs):
        train_idxs = fog_train_json_data[f"{fid}"]
        test_idxs = fog_test_json_data[f"{fid}"]
        classes = list(np.unique(y_train[train_idxs]))
        list_labels_idxes = {k: [] for k in classes}
        for idx in train_idxs:
            list_labels_idxes[y_train[idx]].append(idx)
        list_test_labels_idxes = {k: [] for k in classes}
        for idx in test_idxs:
            list_test_labels_idxes[y_test[idx]].append(idx)
        if partitions == "iid":
            train_json_data = create_iid(
                labels=y_train,
                num_parties=num_parties,
                classes=classes,
                list_labels_idxes=list_labels_idxes,
            )
        elif partitions >= "noniid-label1" and partitions <= "noniid-label9":
            train_json_data, _ = create_noniid(
                train_labels=y_train,
                test_labels=y_test,
                num_parties=num_parties,
                num_classes=int(partitions[-1]),
                classes=classes,
                list_train_labels_idxes=list_labels_idxes,
                list_test_labels_idxes=list_test_labels_idxes
            )
        elif partitions[:10] == "noniid-dir":
            train_json_data, dirichlet_dist = create_noniid_dir(
                labels = y_train,
                num_class=10,
                dirichlet_dist = None,
                num_parties = num_parties,
                alpha = float(partitions[10:]),
                seed = args.seed,
                classes=classes,
                list_labels_idxes=list_labels_idxes)
        write_json(train_json_data, save_dir=save_dir, file_name=f"fid{fid}_train")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
