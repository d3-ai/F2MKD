import json
import timeit
import ray
import concurrent.futures
import torch
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List, cast
from logging import DEBUG, INFO

from flwr.common.logger import log

from common.typing import (
    EvaluateRes,
    Parameters,
    Scalar,
    Weights,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    FitIns,
    FitRes,
    EvaluateIns, 
    EvaluateRes,
    Disconnect,
    Reconnect,
)
from common.parameter import (
    parameters_to_weights,
    weights_to_parameters,
)
from driver import test_parameters, train_kd_multiple, train_kd_parameters, test
from utils.utils_model import load_model
from utils.utils_dataset import load_dataset

from server_app.client_manager import ClientManager, SimpleClientManager
from server_app.client_proxy import ClientProxy
from hfl_server_app.fog_proxy import FogServerProxy
from fog_app.strategy.strategy import FogServerStrategy
from simulation.ray_transport.ray_client_proxy import RayClientProxy

from .strategy.fedmkd import FedMKD

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]

class MKDFogServerProxy(FogServerProxy):
    def __init__(self, fid: str, config: Dict[str, Any], initial_parameters) -> None:
        self.fid = fid
        self.config = config
        self.data_dir = config["data_dir"]
        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]
        self.client_pool = config["client_pool"]
        self.client_partitions = config["client_partitions"]
        self.client_parameters_dict: Dict[str, Parameters] = {
            str(cid): initial_parameters for cid in range(self.client_pool)
        }

        if self.dataset_name[-5:] == "MNIST":
                self.input_spec = (1,28,28)
        elif self.dataset_name == "CIFAR10":
            self.input_spec = (3,32,32)
        else:
            raise NotImplementedError(f"{self.dataset_name} is not Implemented")
        
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.model = load_model(self.model_name, self.input_spec)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_json_path = Path(self.data_dir) / "train.json"
        self.test_json_path = Path(self.data_dir) / "test.json"
        self.train_dataidxs, self.test_dataidxs = self.get_dataidxs()

        # configure the fogserver strategy 
        self._client_manager = SimpleClientManager()
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        
        self.strategy_config = config["strategy_config"]
        self.strategy: FogServerStrategy = FedMKD(
            fraction_fit=self.strategy_config["fraction_fit"],
            min_available_clients=self.strategy_config["min_available_clients"],
            on_fit_config_fn=self.strategy_config["on_fit_config_fn"]   
        )

        self._get_client_config()

        cids = [str(x) for x in range(config["client_pool"])]
        for cid in cids:
            client_proxy = RayClientProxy(
                client_fn=self.client_config["client_fn"],
                cid=cid,
                resources=self.client_config["client_resources"],
                config=self.client_config,
            )
            self.client_manager().register(client=client_proxy)
        

    def set_strategy(self, strategy: FogServerStrategy):
        self.strategy = strategy

    def client_manager(self)->ClientManager:
        return self._client_manager
    
    def get_dataidxs(self):
        with open(self.train_json_path, "r") as f:
            train_dict = json.load(f)
        with open(self.test_json_path, "r") as f:
            test_dict = json.load(f)
        return train_dict[self.fid], test_dict[self.fid]
    
    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        return self.properties
    
    def get_parameters(self) -> ParametersRes:
        weights: Weights = self.model.get_weights()
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    
    def set_parameters(self, parameters) -> None:
        weights: Weights = parameters_to_weights(parameters)
        self.model.set_weights(weights)

    def fit(self, ins: FitIns)->FitRes:# parameters, config) -> FitRes:
        log(INFO,"fit() on fogserver fid=%s",self.fid,)
        parameters = ins.parameters
        config = ins.config
        self.set_parameters(parameters)

        rnd = int(config["epoch_global"])
        batch_size = int(config["batch_size"])
        global_epochs = int(config["global_epochs"])
        client_epochs = int(config["client_epochs"])
        lr=float(config["lr"])
        alpha=float(config["alpha"])
        beta=float(config["beta"])

        clients_kd_config = {
            "fid": self.fid,
            "train_json": self.train_json_path,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "batch_size": batch_size,
            "epochs": client_epochs,
            "lr": lr,
            "alpha": alpha,
        }
        
        results, failures = kd_from_server(ins.parameters, self.client_parameters_dict, clients_kd_config)
        log(
            INFO,
            "train_kd_from_server() on fid=%s: FedMKD received %s results and %s failures",
            self.fid,
            len(results),
            len(failures),
        )
        if len(failures)>0:
            raise ValueError("kd failed")
        for cid, parameters in results:
            self.client_parameters_dict[cid] = parameters 
        
        client_instructions: List[Tuple[ClientProxy, FitIns]] = self.strategy.configure_fit(
            rnd=config["epoch_global"],
            client_parameters_dict=self.client_parameters_dict,
            client_manager=self._client_manager,
        )
        log(
            DEBUG,
            "fit_round on fid=%s: FedMKD sampled %s clients (out of %s)",
            self.fid,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        results, failures = self._fit_clients(client_instructions)
        log(
            DEBUG,
            "fit_round on fid%s: FedMKD received %s results and %s failures",
            self.fid,
            len(results),
            len(failures),
        )

        for client,fit_res in results:
            self.client_parameters_dict[client.cid] = fit_res.parameters

        log(
            INFO,
            "train_kd_from_clients() on fogserver fid=%s",
            self.fid
        )
        trainset = load_dataset(dataset_name=self.dataset_name, dataidxs=self.train_dataidxs, train=True)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        
        self.model.to(self.device)
        train_kd_multiple(
            parameters_dict=self.client_parameters_dict,
            student_net=self.model,
            model_name=self.model_name,
            input_spec=self.input_spec,
            trainloader=trainloader,
            epochs=global_epochs,
            lr=lr,
            alpha=beta,
            device=self.device)
        
        fog_parameters = self.get_parameters().parameters

        return FitRes(
            parameters=fog_parameters,
            num_examples=len(trainset),
            metrics={}
        )
        
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # log(INFO,"eval() on fogserver fid=%s",self.fid,)
        eval_config = {
            "id": self.fid,
            "test_json": self.test_json_path,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "batch_size": 1000,
        }
        results, failures = evaluate_client_parameters(self.client_parameters_dict, eval_config)
        log(
            DEBUG,
            "eval_round on fid=%s received %s results and %s failures",
            self.fid,
            len(results),
            len(failures),
        )
        metrics={}
        for id, res in results:
            loss, accuracy = res
            cid = int(id) + int(self.fid)* int(self.client_pool)
            metrics[str(cid)] = float(accuracy)
        eval_res = EvaluateRes(
            loss=float(loss),
            num_examples=1000,
            metrics=metrics,
        )
        return eval_res
    
    def reconnect(self, reconnect: Reconnect)->Disconnect:
        return Disconnect(reason="")
    
    def _get_client_config(self):
        self.client_config = {}
        self.client_config["dataset_name"] = self.config["client_config"]["dataset_name"]
        self.client_config["model_name"] = self.config["client_config"]["model_name"]
        self.client_config["client_fn"] = self.config["client_config"]["client_fn"]
        self.client_config["client_resources"] = self.config["client_config"]["client_resources"]
        self.client_config["data_dir"] = Path(self.data_dir) / f"C_{self.client_pool}_{self.client_partitions}"
        self.client_config["parent_fid"] = str(self.fid)

    def _get_initial_parameters(self)->Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by fog strategy")
            return parameters
        
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random clietn")
        return parameters_res.parameters
    
    def disconnect_all_clients(self)->None:
        all_clients = self._client_manager.all()
        _ = self._shutdown(clients=[all_clients[k] for k in all_clients.keys()])
    
    def _shutdown(self, clients: List[ClientProxy])->ReconnectResultsAndFailures:
        reconnect = Reconnect(seconds=None)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._reconnect_client, client, reconnect) for client in clients
            ]
            concurrent.futures.wait(futures)

        results: List[Tuple[ClientProxy, Disconnect]] = []
        failures: List[BaseException] = []
        for future in futures:
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
            else:
                result = future.result()
                results.append(result)
        return results, failures
        
    
    def reconnect(self, reconnect: Reconnect)->Disconnect:
        return Disconnect(reason="")
    
    def _get_client_config(self):
        self.client_config = {}
        self.client_config["dataset_name"] = self.config["client_config"]["dataset_name"]
        self.client_config["model_name"] = self.config["client_config"]["model_name"]
        self.client_config["client_fn"] = self.config["client_config"]["client_fn"]
        self.client_config["client_resources"] = self.config["client_config"]["client_resources"]
        self.client_config["data_dir"] = Path(self.data_dir) / f"C_{self.client_pool}_{self.client_partitions}"
        self.client_config["parent_fid"] = str(self.fid)

    def _get_initial_parameters(self)->Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by fog strategy")
            return parameters
        
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random clietn")
        return parameters_res.parameters
    
    def disconnect_all_clients(self)->None:
        all_clients = self._client_manager.all()
        _ = self._shutdown(clients=[all_clients[k] for k in all_clients.keys()])
    
    def _shutdown(self, clients: List[ClientProxy])->ReconnectResultsAndFailures:
        reconnect = Reconnect(seconds=None)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._reconnect_client, client, reconnect) for client in clients
            ]
            concurrent.futures.wait(futures)
        
        results: List[Tuple[ClientProxy, Disconnect]] = []
        failures: List[BaseException] = []
        for future in futures:
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
            else:
                result = future.result()
                results.append(result)
        return results, failures

    def _reconnect_client(
        self,
        client: ClientProxy,
        reconnect: Reconnect)->Tuple[ClientProxy, Disconnect]:
        disconnect = client.reconnect(reconnect)
        return client, disconnect
    
    def _fit_clients(
        self,
        client_instructions: List[Tuple[ClientProxy, FitIns]])->FitResultsAndFailures:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._fit_client, client, ins) for client, ins in client_instructions
            ]
            concurrent.futures.wait(futures)

        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[BaseException] = []
        for future in futures:
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
            else:
                result = future.result()
                results.append(result)
        return results, failures

    def _fit_client(
        self,
        client: ClientProxy,
        ins: FitIns)->Tuple[ClientProxy, FitRes]:
        fit_res = client.fit(ins)
        return client, fit_res

    def _evaluate_clients(
        self,
        client_instructions: List[Tuple[ClientProxy, EvaluateIns]])->EvaluateResultsAndFailures:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_client, client, ins) for client, ins in client_instructions
            ]
            concurrent.futures.wait(futures)
        
        results: List[Tuple[ClientProxy, EvaluateRes]] =[]
        failures: List[BaseException] = []
        for future in futures:
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
            else:
                result = future.result()
                results.append(result)
        return results, failures
        

    def _evaluate_client(
        self,
        client: ClientProxy,
        ins: FitIns)->Tuple[ClientProxy, FitRes]:
        evaluate_res = client.evaluate(ins)
        return client, evaluate_res
    
def kd_from_server(
    server_parameters: Parameters,
    client_parameters_dict: Dict[str, Parameters],
    config: Dict[str, Any]):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(kd_parameters, cid, server_parameters, client_parameters, config) for cid, client_parameters in client_parameters_dict.items()
        ]
        concurrent.futures.wait(futures)

    results: List[Tuple[str, Parameters]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def kd_parameters(
    cid: str,
    server_parameters: Parameters,
    client_parameters: Parameters,
    config: Dict[str, Any]):
    future_kd_res = train_kd_parameters.remote(
        server_parameters, client_parameters, config
    )
    res = ray.worker.get(future_kd_res)
    return cid, cast(Parameters, res)

def evaluate_client_parameters(
    client_parameters_dict: Dict[str, Parameters],
    config: Dict[str, Any]):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_parameters, cid, client_parameters, config) for cid, client_parameters in client_parameters_dict.items()
        ]
        concurrent.futures.wait(futures)

    results: List[Tuple[str, Tuple[float,float]]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def evaluate_parameters(
    cid: str,
    client_parameters: Parameters,
    config: Dict[str, Any]):
    
    future_eval_res = test_parameters.remote(
        client_parameters, config
    )
    res = ray.worker.get(future_eval_res)
    loss, acc = cast(Tuple[float,float], res) 
    return cid, (loss, acc)
