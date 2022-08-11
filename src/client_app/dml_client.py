import json
import ray
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from logging import INFO, DEBUG
from flwr.common.logger import log

from typing import Dict, Any, Tuple, cast
from common.parameter import parameters_to_weights, weights_to_parameters

from common.typing import (
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    Weights,
    PropertiesIns,
    PropertiesRes,
    ParametersRes,
    FitIns,
    FitRes,
    Reconnect,
    Disconnect,
)
from server_app.client_proxy import ClientProxy
from driver import loss_kd, test_parameters
from utils.utils_model import load_model
from utils.utils_dataset import load_dataset

class DMLClientProxy(ClientProxy):
    def __init__(self, cid: str, initial_parameters: Parameters, config: Dict[str, str]) -> None:
        self.cid = cid
        self.data_dir = config["data_dir"]
        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]
        self.partitions = config["client_partitions"]
        self.client_pool = int(config["client_pool"])
        self.parent_fid =str(int(cid) // self.client_pool)
        self.parameters = initial_parameters
        
        if self.dataset_name[-5:] == "MNIST":
                self.input_spec = (1,28,28)
        elif self.dataset_name == "CIFAR10":
            self.input_spec = (3,32,32)
        else:
            raise NotImplementedError(f"{self.dataset_name} is not Implemented")

        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        
        self.train_json_path = Path(self.data_dir) / f"C_{self.client_pool}_{self.partitions}" / f"fid{self.parent_fid}_train.json"
        self.test_json_path = Path(self.data_dir) / "test.json"

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        return self.get_properties

    def get_parameters(self) -> ParametersRes:
        return self.parameters
    
    def fit(self, ins: FitIns)->FitRes:
        log(INFO,"fit() on fid=%s client cid=%s", self.parent_fid, self.cid,)
        config = ins.config
        fit_config = {
            "id": str(int(self.cid) % self.client_pool),
            "train_json": self.train_json_path,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "batch_size": 1000,
            "epochs": int(config["epochs"]),
            "lr": float(config["lr"]),
            "alpha": float(config["alpha"]),
            "beta": float(config["beta"]),
        }

        future_res = train_mutual.remote(
            ins.parameters, self.parameters, fit_config
        )
        res = ray.worker.get(future_res)
        meme_parameters, self.parameters = cast(
            Tuple[Parameters, Parameters], res
        )

        return FitRes(
            parameters=meme_parameters,
            num_examples=60,
        )
    
    def evaluate(self, ins: EvaluateIns)->EvaluateRes:
        eval_config = {
            "id": self.parent_fid,
            "test_json": self.test_json_path,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "batch_size": 1000
        }
        future_res = test_parameters.remote(
            self.parameters, eval_config
        )
        res = ray.worker.get(future_res)
        loss, accuracy = cast(Tuple[float, float], res)

        metrics = {
            str(self.cid): float(accuracy)
        }
        eval_res = EvaluateRes(
            loss=float(loss),
            num_examples=1000,
            metrics=metrics
        )
        return eval_res
    def reconnect(self, reconnect: Reconnect) -> Disconnect:
        """Disconnect and (optionally) reconnect later."""
        return Disconnect(reason="")  # Nothing to do here (yet)

@ray.remote(num_cpus=1)
def train_mutual(
    meme_parameters: Parameters,
    client_parameters: Parameters,
    config: Dict[str, Any])->Tuple[Parameters, Parameters]:
    id = config["id"]
    train_json = config["train_json"]
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    alpha = config["alpha"]
    beta = config["beta"]
    if dataset_name[-5:] == "MNIST":
            input_spec = (1,28,28)
    elif dataset_name == "CIFAR10":
        input_spec = (3,32,32)
    else:
        raise NotImplementedError(f"{dataset_name} is not Implemented")
    with open(train_json, "r") as f:
        train_dict = json.load(f)
    
    num_workers = len(ray.worker.get_resource_ids()["CPU"])
    trainset = load_dataset(dataset_name=dataset_name, train=True, dataidxs=train_dict[id])
    if torch.cuda.is_available():
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, shuffle=True)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    meme_model = load_model(model_name=model_name, input_spec=input_spec)
    meme_model.set_weights(parameters_to_weights(meme_parameters))
    client_model = load_model(model_name=model_name, input_spec=input_spec)
    client_model.set_weights(parameters_to_weights(client_parameters))

    meme_optimizer = torch.optim.SGD(meme_model.parameters(), lr=lr)
    client_optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meme_model.to(device)
    client_model.to(device)
    for _ in range(epochs):
        meme_model.eval()
        client_model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                meme_outputs = meme_model(images)
            client_outputs = client_model(images)
            client_optimizer.zero_grad()
            loss = loss_kd(client_outputs ,labels, meme_outputs, alpha)
            loss.backward()
            client_optimizer.step()
        client_model.eval()
        meme_model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                client_outputs = client_model(images)
            meme_outputs = meme_model(images)
            meme_optimizer.zero_grad()
            loss = loss_kd(meme_outputs ,labels, client_outputs, beta)
            loss.backward()
            meme_optimizer.step()
    
    return (weights_to_parameters(meme_model.get_weights()), weights_to_parameters(client_model.get_weights()))
        