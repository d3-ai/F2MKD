import argparse
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
parser.add_argument("--dataset", type=str, required=True, choices=["FashionMNIST","MNIST","CIFAR10"], help="dataset name")
parser.add_argument("--partitions", type=str, required=True, help="dataset partitions")
parser.add_argument("--save_dir", type=str, required=True, help="save directory")
parser.add_argument("--seed", type=int, required=True, help="random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    set_seed(args.seed)
    num_parties = args.num_fogs
    dataset = args.dataset
    partitions = args.partitions
    save_dir = Path(args.save_dir)
    if dataset == "MNIST":
        _, y_train, _, y_test = load_mnist()
    elif dataset == "FashionMNIST":
        _, y_train, _, y_test = load_fmnist()
    elif dataset == "CIFAR10":
        _, y_train, _, y_test = load_cifar10()
    else:
        raise NotImplementedError(f"{dataset} is no implemented")
    
    
    if partitions == "iid":
        train_json_data = create_iid(
            labels=y_train,
            num_parties=num_parties,
        )
        test_json_data = create_iid(
            labels=y_test,
            num_parties=num_parties,
        )
    elif partitions >= "noniid-label1" and partitions <= "noniid-label9":
        train_json_data, test_json_data = create_noniid(
            train_labels=y_train,
            test_labels=y_test,
            num_parties=num_parties,
            num_classes=int(partitions[-1]),
        )
    elif partitions[:10] == "noniid-dir":
        train_json_data, dirichlet_dist = create_noniid_dir(
            labels = y_train,
            num_class=10,
            dirichlet_dist = None,
            num_parties = num_parties,
            alpha = float(partitions[10:]),
            seed = args.seed)
        test_json_data, dirichlet_dist = create_noniid_dir(
            labels = y_test,
            num_class=10,
            dirichlet_dist = dirichlet_dist,
            num_parties = num_parties,
            alpha = float(partitions[10:]),
            seed = args.seed)
    write_json(train_json_data, save_dir=save_dir, file_name="train")
    write_json(test_json_data, save_dir=save_dir, file_name="test")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
