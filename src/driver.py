import json
import ray
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logging import DEBUG
from flwr.common.logger import log
from common.typing import Parameters
from common.parameter import (
    parameters_to_weights,
    weights_to_parameters,
)
from models.net import Net
from utils.utils_model import load_model
from utils.utils_dataset import load_dataset

def train(net: Net, trainloader: DataLoader, epochs: int, lr: float, device: str):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def train_mutual(
    client_net: Net,
    meme_net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    device:str):
    client_optimizer = torch.optim.SGD(client_net.parameters(),lr=lr)
    meme_optimizer = torch.optim.SGD(meme_net.parameters(), lr=lr)
    client_net.to(device)
    meme_net.to(device)
    for _ in range(epochs):
        meme_net.eval()
        client_net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                meme_outputs = meme_net(images)
            client_outputs = client_net(images)
            client_optimizer.zero_grad()
            loss = loss_kd(client_outputs, labels, meme_outputs, alpha)
            loss.backward()
            client_optimizer.step()
        client_net.eval()
        meme_net.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                client_outputs = client_net(images)
            meme_outputs = meme_net(images)
            meme_optimizer.zero_grad()
            loss = loss_kd(meme_outputs, labels, client_outputs, beta)
            loss.backward()
            meme_optimizer.step()


@ray.remote
def train_kd_parameters(
    server_parameters: Parameters,
    client_parameters: Parameters,
    config: Dict[str, Any])->Parameters:
    fid = config["fid"]
    train_json = config["train_json"]
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    alpha = config["alpha"]
    if dataset_name[-5:] == "MNIST":
            input_spec = (1,28,28)
    elif dataset_name == "CIFAR10":
        input_spec = (3,32,32)
    else:
        raise NotImplementedError(f"{dataset_name} is not Implemented")
    with open(train_json, "r") as f:
        train_dict = json.load(f)
    trainset = load_dataset(dataset_name=dataset_name, train=True, dataidxs=train_dict[fid])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    teacher_model = load_model(model_name=model_name, input_spec=input_spec)
    teacher_model.set_weights(parameters_to_weights(server_parameters))
    for param in teacher_model.parameters():
        param.grad = None
    
    student_model = load_model(model_name=model_name, input_spec=input_spec)
    student_model.set_weights(parameters_to_weights(client_parameters))
    optimizer = torch.optim.SGD(student_model.parameters(),lr =lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher_model.to(device)
    teacher_model.eval()
    student_model.to(device)
    student_model.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels  = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            optimizer.zero_grad()
            loss = loss_kd(student_outputs ,labels, teacher_outputs, alpha)
            loss.backward()
            optimizer.step()
            break
    return weights_to_parameters(student_model.get_weights())
    


def train_kd_multiple(
    parameters_dict: Dict[str, Parameters],
    student_net: Net,
    model_name: str,
    input_spec: List,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    alpha: float,
    device: torch.device)->Parameters:
    teacher_net_list: List[Net] = []
    for _, parameters in parameters_dict.items():
        net = load_model(model_name, input_spec)
        net.set_weights(parameters_to_weights(parameters))
        teacher_net_list.append(net.to(device).eval())
    for teacher_net in teacher_net_list:
        for param in teacher_net.parameters():
            param.grad = None
    optimizer = torch.optim.SGD(student_net.parameters(), lr=lr)
    student_net.to(device)
    student_net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs_list = [net(images) for net in teacher_net_list]
            student_outputs = student_net(images)
            loss = loss_kd_multiple(student_outputs, labels, teacher_outputs_list, alpha)
            loss.backward()
            optimizer.step()
            break


def loss_kd(outputs, labels, teacher_outputs, alpha):
    loss = alpha*nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs, dim=1), F.softmax(teacher_outputs, dim=1)) + (1-alpha) * F.cross_entropy(outputs, labels)
    return loss

def loss_kd_multiple(outputs, labels, teacher_outputs_list, alpha):
    loss = (1-alpha) * F.cross_entropy(outputs, labels)
    for teacher_outputs in teacher_outputs_list:
        loss += alpha*nn.KLDivLoss(reduction="batchmean")(F.log_softmax(outputs, dim=1), F.softmax(teacher_outputs, dim=1)) / len(teacher_outputs_list)
    return loss

def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

@ray.remote(num_cpus=1)
def test_parameters(parameters, config):
    """Validate the network on the entire test set."""
    id = config["id"]
    test_json = config["test_json"]
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    batch_size = config["batch_size"]
    if dataset_name[-5:] == "MNIST":
            input_spec = (1,28,28)
    elif dataset_name == "CIFAR10":
        input_spec = (3,32,32)
    else:
        raise NotImplementedError(f"{dataset_name} is not Implemented")
    with open(test_json, "r") as f:
        test_dict = json.load(f)
    testset = load_dataset(dataset_name=dataset_name, train=False, dataidxs=test_dict[id])
    testloader = DataLoader(testset, batch_size=batch_size,pin_memory=True)
    net = load_model(model_name=model_name, input_spec=input_spec)
    net.set_weights(parameters_to_weights(parameters))
    for param in net.parameters():
        param.grad = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy