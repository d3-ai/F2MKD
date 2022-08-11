import json
import os
import random
import torch
import numpy as np
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import transforms

from typing import List, Dict, Tuple
from pathlib import Path

from utils.utils_dataset import CIFAR10_truncated
DATA_ROOT = Path("./data")

def load_fmnist():
    transform = transforms.Compose([transforms.ToTensor()])

    traindata = FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    testdata = FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    X_train, y_train = traindata.data, traindata.targets
    X_test, y_test = testdata.data, testdata.targets

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    return (X_train, y_train, X_test, y_test)

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])

    traindata = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    testdata = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    X_train, y_train = traindata.data, traindata.targets
    X_test, y_test = testdata.data, testdata.targets

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    return (X_train, y_train, X_test, y_test)

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    root = DATA_ROOT / "CIFAR10" / "raw"

    traindata = CIFAR10_truncated(root=root, train=True, download=True, transform=transform)
    testdata = CIFAR10_truncated(root=root, train=False, download=True, transform=transform)

    X_train, y_train = traindata.data, traindata.target
    X_test, y_test = testdata.data, testdata.target
    return (X_train, y_train, X_test, y_test)

def create_iid(
    labels: np.ndarray,
    num_parties: int,
    classes: List[int] = None,
    list_labels_idxes: Dict[int, List[int]] = None):
    if labels.shape[0] % num_parties:
        raise ValueError("Imbalanced classes are not allowed")

    if classes is None and list_labels_idxes is None:
        print("creating label_idxes ...")
        classes = list(np.unique(labels))
        list_labels_idxes = {k: np.where(labels == k)[0].tolist() for k in classes}
    elif classes is None or list_labels_idxes is None:
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_labels_idxes = list_labels_idxes
    
    net_dataidx_map = {i: [] for i in range(num_parties)}
    id = 0
    for k in classes:
        while(len(list_labels_idxes[k]) > 0):
            label_idx = list_labels_idxes[k].pop()
            net_dataidx_map[id % num_parties].append(label_idx)
            id += 1
    record_net_data_stats(labels,net_dataidx_map)
    return net_dataidx_map

def create_noniid(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    num_parties: int,
    num_classes: int,
    classes: List[int] = None,
    list_train_labels_idxes: Dict[int, List[int]] = None,
    list_test_labels_idxes: Dict[int, List[int]] = None):
    if train_labels.shape[0] % (num_parties*num_classes):
        raise ValueError("Imbalanced classes are not allowed")

    if classes is None and list_train_labels_idxes is None and list_test_labels_idxes is None:
        print("creating label_idxes ...")
        classes = list(np.unique(train_labels))
        list_train_labels_idxes = {k: np.where(train_labels == k)[0].tolist() for k in classes}
        list_test_labels_idxes = {k: np.where(test_labels == k)[0].tolist() for k in classes}
        train_samples_per_class = int(train_labels.shape[0] / (num_parties * num_classes))
        test_samples_per_class = int(test_labels.shape[0] / (num_parties * num_classes))
    elif classes is None or list_train_labels_idxes is None or list_test_labels_idxes is None:
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_train_labels_idxes = list_train_labels_idxes
        num_train = 0
        for val in list_train_labels_idxes.values():
            num_train += len(val)
        list_test_labels_idxes = list_test_labels_idxes
        num_test = 0
        for val in list_test_labels_idxes.values():
            num_test += len(val)
        train_samples_per_class = int(num_train / (num_parties * num_classes))
        test_samples_per_class = int(num_test / (num_parties * num_classes))
    
    train_json_data = {i: [] for i in range(num_parties)}
    test_json_data = {i: [] for i in range(num_parties)}

    class_ids = list(np.random.permutation(classes))
    for id in range(num_parties):
        for i in range(num_classes):
            cls = class_ids.pop()
            for _ in range(train_samples_per_class):
                train_idx = list_train_labels_idxes[cls].pop()
                train_json_data[id].append(train_idx)
            for _ in range(test_samples_per_class):
                test_idx = list_test_labels_idxes[cls].pop()
                test_json_data[id].append(test_idx)
            if len(class_ids) == 0:
                class_ids = list(np.random.permutation(classes))
    
    record_net_data_stats(train_labels,train_json_data)
    record_net_data_stats(test_labels,test_json_data)
    
    return train_json_data, test_json_data


def create_noniid_dir(
    labels: np.ndarray,
    num_class: int,
    dirichlet_dist: np.ndarray,
    num_parties: int,
    alpha: float,
    seed: int,
    classes: List[int] = None,
    list_labels_idxes: Dict[int, List[int]] = None):

    if labels.shape[0] % num_parties:
        raise ValueError("Imbalanced classes are not allowed")
    
    
    if classes is None and list_labels_idxes is None:
        print("creating label_idxes ...")
        classes = list(np.unique(labels))
        list_labels_idxes = {k: np.where(labels == k)[0].tolist() for k in classes}
        num_samples = [int(labels.shape[0] / num_parties) for _ in range(num_parties)]
    elif classes is None or list_labels_idxes is None:
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_labels_idxes = list_labels_idxes
        num_sample = 0
        for val in list_labels_idxes.values():
            num_sample += len(val)
        num_samples = [int(num_sample / num_parties) for _ in range(num_parties)]
    alpha = np.asarray(alpha)
    alpha = np.repeat(alpha, num_class)
    
    if dirichlet_dist is None:
        dirichlet_dist =  np.random.default_rng(seed).dirichlet(
            alpha=alpha, size=num_parties 
        )
        if dirichlet_dist.size != 0:
            if dirichlet_dist.shape != (num_parties, num_class):
                raise ValueError("The shape of the given dirichlet distribution is no allowed")
    
    empty_classes =[False if i in classes else True for i in range(num_class)]
    print(empty_classes)

    net_dataidx_map = {i: [] for i in range(num_parties)}
    for id in range(num_parties):
        net_dataidx_map[id], empty_classes = sample_without_replacement(
            distribution=dirichlet_dist[id].copy(),
            list_label_idxes=list_labels_idxes,
            num_sample=num_samples[id],
            empty_classes=empty_classes,
        )

    record_net_data_stats(labels, net_dataidx_map)
    return net_dataidx_map, dirichlet_dist

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print(str(net_cls_counts))
    return net_cls_counts

def sample_without_replacement(
    distribution: np.ndarray,
    list_label_idxes: Dict[int, List[np.ndarray]],
    num_sample: int,
    empty_classes: List[bool])-> Tuple[List[int], List[bool]]:
    distribution = exclude_classes_and_normalize(
        distribution=distribution,
        exclude_dims=empty_classes,
    )
    label_list = []
    for _ in range(num_sample):
        sample_class = np.where(np.random.multinomial(1, distribution)==1)[0][0]
        label_idx = list_label_idxes[sample_class].pop()
        label_list.append(label_idx)
        if len(list_label_idxes[sample_class]) == 0:
            empty_classes[sample_class] = True
            distribution = exclude_classes_and_normalize(
                distribution=distribution,
                exclude_dims=empty_classes,
            )
    np.random.shuffle(label_list)
    return label_list, empty_classes 


def exclude_classes_and_normalize(
    distribution: np.ndarray,
    exclude_dims: List[bool],
    eps: float = 1e-5)-> np.ndarray:
    if np.any(distribution < 0) or (not np.isclose(np.sum(distribution), 1.0)):
        raise ValueError("distribution must sum to 1 and have only positive values.")

    if distribution.size != len(exclude_dims):
        raise ValueError(
            """Length of distribution must be equal
            to the length `exclude_dims`."""
        )
    if eps < 0:
        raise ValueError("""The value of `eps` must be positive and small.""")
    
    distribution[[not x for x in exclude_dims]] += eps
    distribution[exclude_dims] = 0.0
    sum_rows = np.sum(distribution) + np.finfo(float).eps
    distribution = distribution / sum_rows
    return distribution

def write_json(json_data: Dict[str, List[np.ndarray]], save_dir: str, file_name: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_path = Path(save_dir) / f"{file_name}.json"
    print("writing {}".format(file_name))
    with open(file_path, "w") as outfile:
        json.dump(json_data, outfile)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    dataset = "FashionMNIST"
    alpha = 0.5
    X_train, y_train, X_test, y_test = load_fmnist()
    set_seed(1234)
    net_dataidx_map, dirichlet_dist = create_noniid_dir(
        labels = y_train,
        num_class=10,
        dirichlet_dist = None,
        num_parties = 1000,
        alpha = alpha,
        seed = 1234)
    write_json(net_dataidx_map, dataset, file_name="noniid-dir", train=True)
    net_dataidx_map = create_iid(
        labels=y_train,
        num_parties=50,
    )
    write_json(net_dataidx_map, dataset, file_name="iid", train=True)

    # net_dataidx_map, dirichlet_dist = create_noniid_dir(
    #     labels = y_test,
    #     dirichlet_dist = dirichlet_dist,
    #     num_parties = 10,
    #     alpha = alpha,
    #     seed = 1234)
    # write_json(net_dataidx_map, dataset, "test_data")
