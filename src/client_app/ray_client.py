import json
import timeit
import ray
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from logging import INFO, DEBUG
from flwr.common.logger import log

from typing import Dict
from common.parameter import parameters_to_weights, weights_to_parameters

from common.typing import (
    EvaluateIns,
    EvaluateRes,
    Scalar,
    Weights,
    PropertiesIns,
    PropertiesRes,
    ParametersRes,
    FitIns,
    FitRes,
)
from .client import Client
from driver import train, train_mutual, test
from utils.utils_model import load_model
from utils.utils_dataset import load_dataset


class RayClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]) -> None:
        self.cid = cid
        self.data_dir = config["data_dir"]
        self.dataset_name = config["dataset_name"]
        if self.dataset_name == "FashionMNIST" or self.dataset_name == "MNIST":
            self.input_spec = (1,28,28)
        else:
            raise NotImplementedError(f"{self.dataset_name} is not Implemented")
        self.model_name = config["model_name"]

        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = load_model(model_name=self.model_name, input_spec=self.input_spec)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dataidxs = self.get_dataidxs()
    
    def get_dataidxs(self):
        fid = int(self.cid) // 200
        json_path = Path(self.data_dir) / f"fid{fid}_train.json"
        with open(json_path, "r") as f:
            dict = json.load(f)
        cid = int(self.cid) % 200
        return dict[str(cid)]

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        return self.get_properties

    def get_parameters(self) -> ParametersRes:
        weights: Weights = self.net.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    
    def set_parameters(self, parameters)->None:
        weights: Weights = parameters_to_weights(parameters)
        self.net.set_weights(weights)

    def fit(self, ins: FitIns)->FitRes:
        log(INFO,"fit() on client cid=%s", self.cid,)
        parameters = ins.parameters
        config = ins.config
        self.set_parameters(parameters)

        batch_size = int(config["batch_size"])
        epochs = int(config["epochs"])
        lr = float(config["lr"])

        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = load_dataset(dataset_name=self.dataset_name, dataidxs=self.dataidxs, train=True)
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,persistent_workers=True, pin_memory=True, shuffle=True)

        self.net.to(self.device)
        train(self.net, trainloader, epochs=epochs, lr=lr, device=self.device)
        return FitRes(
            parameters=self.get_parameters().parameters, num_examples=len(trainloader.dataset),
        )
    
    def evaluate(self, ins: EvaluateIns)->EvaluateRes:
        print(f"evaluate() on client cid={self.cid}")
        raise NotImplementedError("Client evaluation code is not implemented")
        # parameters = ins.parameters
        # config = ins.config
        # self.set_parameters(parameters)
        
        # num_workers = len(ray.worker.get_resource_ids()["CPU"])
        # _,testset,_ = load_partitions(self.cid, self.data_dir)
        # testloader = DataLoader(testset, batch_size=int(config["batch_size"]), num_workers=num_workers)

        # self.model.to(self.device)
        # loss, accuracy = test(self.model, testloader, epochs=int(config["epochs"]), device=self.device)
        # return EvaluateRes(
        #     num_examples=len(testset), loss=float(loss), metrics={"accuracy": float(accuracy)}
        # )

class RayDMLClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]) -> None:
        self.cid = cid
        self.data_dir = config["data_dir"]
        self.test_json = config["test_json"]
        self.dataset_name = config["dataset_name"]
        
        if self.dataset_name[-5:] == "MNIST":
                self.input_spec = (1,28,28)
        elif self.dataset_name == "CIFAR10":
            self.input_spec = (3,32,32)
        else:
            raise NotImplementedError(f"{self.dataset_name} is not Implemented")
        self.model_name = config["model_name"]

        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.client_net = load_model(model_name=self.model_name, input_spec=self.input_spec)
        self.meme_net = load_model(model_name=self.model_name, input_spec=self.input_spec)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.train_dataidxs, self.test_dataidxs = self.get_dataidxs()
    
    def get_dataidxs(self):
        fid = int(self.cid) // 100
        train_json = Path(self.data_dir) / f"fid{fid}_train.json"
        with open(train_json, "r") as f:
            dict = json.load(f)
        cid = int(self.cid) % 100
        train_dataidxs = dict[str(cid)]
        with open(self.test_json, "r") as f:
            dict = json.load(f)
        test_dataidxs = dict[str(fid)]
        return train_dataidxs, test_dataidxs

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        return self.get_properties

    def get_parameters(self) -> ParametersRes:
        weights: Weights = self.meme_net.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    
    def get_dualparameters(self) -> ParametersRes:
        weights: Weights = self.client_net.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    
    def set_parameters(self, meme_parameters, client_parameters)->None:
        meme_weights: Weights = parameters_to_weights(meme_parameters)
        client_weights: Weights = parameters_to_weights(client_parameters)
        self.meme_net.set_weights(meme_weights)
        self.client_net.set_weights(client_weights)

    def fit(self, ins: FitIns)->FitRes:
        log(INFO,"fit() on client cid=%s",self.cid,)
        meme_parameters = ins.parameters
        client_parameters = ins.dual_parameters
        config = ins.config
        self.set_parameters(meme_parameters, client_parameters)
        batch_size = int(config["batch_size"])
        epochs = int(config["epochs"])
        lr = float(config["lr"])
        alpha = float(config["alpha"])
        beta = float(config["beta"])

        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = load_dataset(dataset_name=self.dataset_name, dataidxs=self.train_dataidxs, train=True)
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, pin_memory=True, shuffle=True)

        self.client_net.to(self.device)
        self.meme_net.to(self.device)
        train_mutual(client_net=self.client_net,meme_net=self.meme_net,trainloader=trainloader,epochs=epochs,lr=lr, alpha=alpha, beta=beta, device=self.device)
        return FitRes(
            parameters=self.get_parameters().parameters,
            dual_parameters=self.get_dualparameters().parameters,
            num_examples=len(trainloader.dataset),
        )
    
    def evaluate(self, ins: EvaluateIns)->EvaluateRes:
        client_parameters = ins.dual_parameters
        self.client_net.set_weights(parameters_to_weights(client_parameters))

        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        testset = load_dataset(dataset_name=self.dataset_name, dataidxs=self.test_dataidxs, train=False)
        
        testloader = DataLoader(testset, batch_size=1000, num_workers=num_workers)

        self.client_net.to(self.device)
        client_loss, client_accuracy = test(self.client_net, testloader, device=self.device)
        metrics = {
            str(self.cid): float(client_accuracy)
        }
        eval_res = EvaluateRes(
            loss=float(client_loss),
            num_examples=int(len(testset)),
            metrics=metrics
        )
        return eval_res


class RayFogClient(Client):
    """Client for hfl /mkd ray simulation"""
    def __init__(self, cid: str, config: Dict[str, str]) -> None:
        self.cid = cid
        self.parent_fid = config["parent_fid"]
        self.data_dir = config["data_dir"]
        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]
        
        if self.dataset_name[-5:] == "MNIST":
                self.input_spec = (1,28,28)
        elif self.dataset_name == "CIFAR10":
            self.input_spec = (3,32,32)
        else:
            raise NotImplementedError(f"{self.dataset_name} is not Implemented")

        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.model = load_model(model_name=self.model_name, input_spec=self.input_spec)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.train_json_path = Path(self.data_dir) / f"fid{self.parent_fid}_train.json"
        self.test_json_path = Path(self.data_dir) / f"fid{self.parent_fid}_test.json"
        self.train_dataidxs = self.get_dataidxs() # , self.test_dataidxs = self.get_dataidxs()
    
    def get_dataidxs(self):
        with open(self.train_json_path, "r") as f:
            train_dict = json.load(f)
        # with open(self.train_json_path, "r") as f:
        #     test_dict = json.load(f)
        return train_dict[self.cid]#, test_dict[self.cid]

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        return self.get_properties

    def get_parameters(self) -> ParametersRes:
        weights: Weights = self.model.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    
    def set_parameters(self, parameters)->None:
        weights: Weights = parameters_to_weights(parameters)
        self.model.set_weights(weights)

    def fit(self, ins: FitIns)->FitRes:
        # log(INFO,"fit() on fid=%s client cid=%s",self.parent_fid, self.cid,)
        parameters = ins.parameters
        config = ins.config
        self.set_parameters(parameters)

        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = load_dataset(dataset_name=self.dataset_name, dataidxs=self.train_dataidxs, train=True)
        if torch.cuda.is_available():
            trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]), num_workers=num_workers, pin_memory=True, persistent_workers=True, shuffle=True)
        else:
            trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]), pin_memory=True, shuffle=True)

        self.model.to(self.device)
        train(self.model, trainloader, epochs=int(config["epochs"]), lr=float(config["lr"]), device=self.device)
        return FitRes(
            parameters=self.get_parameters().parameters, num_examples=len(trainloader.dataset),
        )
    
    def evaluate(self, ins: EvaluateIns)->EvaluateRes:
        print(f"evaluate() on client cid={self.cid}")
        raise NotImplementedError("Client evaluation code is not implemented")
        # parameters = ins.parameters
        # config = ins.config
        # self.set_parameters(parameters)
        
        # num_workers = len(ray.worker.get_resource_ids()["CPU"])
        # _,testset,_ = load_partitions(self.cid, self.data_dir)
        # testloader = DataLoader(testset, batch_size=int(config["batch_size"]), num_workers=num_workers)

        # self.model.to(self.device)
        # loss, accuracy = test(self.model, testloader, epochs=int(config["epochs"]), device=self.device)
        # return EvaluateRes(
        #     num_examples=len(testset), loss=float(loss), metrics={"accuracy": float(accuracy)}
        # )