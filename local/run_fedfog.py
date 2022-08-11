import ray
import gc
import argparse
import random
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from typing import Dict, Any
from pathlib import Path

from common.typing import Weights
from common.parameter import weights_to_parameters, parameters_to_weights
from hfl_server_app.strategy.strategy import HFLStrategy
from simulation.hfl_app import start_simulation
from hfl_server_app.strategy.fedavg import FedAvg
from client_app.ray_client import RayFogClient
from utils.utils_model import load_model
from utils.notify import Line
from driver import test
DATA_ROOT = Path("./data")

parser = argparse.ArgumentParser("FedFog ray simulation using Flower")

parser.add_argument("--num_rounds", type=int, required=True, help="The number of global rounds of federated learning")
parser.add_argument("--fraction_fit", type=float, required=True, help="fraction of the fogservers for each round")
parser.add_argument("--num_fogs", type=int, required=True, help="The number of fogservers")
parser.add_argument("--client_pool", type=int, required=True, help="The number of clients for each fogserver ")
parser.add_argument("--dataset", type=str, required=True, choices=["FashionMNIST","MNIST"], help="dataset name")
parser.add_argument("--client_partitions", type=str, required=True, help="client data partitions")
parser.add_argument("--model_name", type=str, required=True, choices=["mlp", "tinyCNN"], help="Model to train")
parser.add_argument("--client_epochs", type=int, required=True, help="The number of epochs for local training on client")
parser.add_argument("--client_batch_size", type=int, required=True, help="The number of batch_size for local training on client")
parser.add_argument("--lr", type=float, required=True, help="learning rate")
parser.add_argument("--data_dir", type=str, required=True, help="partitioned data directory")
parser.add_argument("--save_dir", type=str, required=True, help="save directory for model params, logs and others")
parser.add_argument("--seed", type=int, required=True, help="random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(args: argparse.ArgumentParser):
    set_seed(args.seed) 
    fog_resources = {"num_cpus": 1}
    client_resources = {"num_cpus": 1}#, "num_gpus": 0.05}
    
    if args.dataset == "FashionMNIST" or args.dataset == "MNIST":
            input_spec = (1,28,28)
    else:
        raise NotImplementedError(f"{args.dataset} is not Implemented")

    init_model = load_model(args.model_name, input_spec)
    init_parameters = weights_to_parameters(init_model.get_weights())

    transform = transforms.Compose([transforms.ToTensor()])
    testset = FashionMNIST(root=DATA_ROOT, train=False, transform=transform)
    
    def client_fn(cid: str, config: Dict[str, Any]):
        return RayFogClient(cid, config)    

    def fog_fit_config(rnd: int)->Dict[str,str]:
        config = {
            "epoch_global": str(rnd),
        }
        return config
    
    def client_fit_config(rnd: int)->Dict[str, str]:
        config = {
            "epoch_global": str(rnd),
            "epochs": str(args.client_epochs),
            "batch_size": str(args.client_batch_size),
            "lr": str(args.lr)
        }
        return config
    
    def get_eval_fn(testset: Dataset):
        def evaluate(weights: Weights):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = load_model(args.model_name, input_spec=input_spec)
            net.set_weights(weights)
            net.to(device)
            testloader = DataLoader(testset, batch_size=1000)
            loss, acc = test(net, testloader, device=device)
            return loss, {"accuracy": acc}
        return evaluate

    client_config: Dict[str, Any]
    client_config = {
        "dataset_name": args.dataset,
        "model_name": args.model_name,
        "client_fn": client_fn,
        "client_resources": client_resources,
    }

    fog_strategy_config: Dict[str, Any]
    fog_strategy_config = {
        "fraction_fit": args.fraction_fit,
        "min_available_clients": args.client_pool,
        "on_fit_config_fn": client_fit_config,
    }

    fog_config: Dict[str, Any]
    fog_config = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset,
        "model_name": args.model_name,
        "client_pool": args.client_pool,
        "client_partitions": args.client_partitions,
        "client_config": client_config,
        "strategy_config": fog_strategy_config,
    }

    # configure the strategy
    server_strategy: HFLStrategy = FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_available_fogs=args.num_fogs,  # All clients should be available
        on_fit_config_fn=fog_fit_config,
        eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
        initial_parameters=init_parameters,
    )


    # (optional) specify ray config
    ray_config = {"include_dashboard": False, "num_cpus": 20}

    hist, params = start_simulation(
        strategy=server_strategy,
        num_fogs=args.num_fogs,
        num_rounds=args.num_rounds,
        fog_config=fog_config,
        fog_resources=fog_resources,
        ray_init_args=ray_config,
    )
    init_model.set_weights(parameters_to_weights(params))
    model_path = Path(args.save_dir) / "models" / "global_model.pth"
    torch.save(init_model.cpu().state_dict(), model_path)
    metrics_path = Path(args.save_dir) / "metrics" / "centralized_global_acc.mat"
    scipy.io.savemat(metrics_path, {"accuracy": np.array(hist.metrics_centralized["accuracy"])})
    for fid, val in hist.metrics_distributed.items():
        metrics_path = Path(args.save_dir) / "metrics" / f"distributed_fid_{fid}_acc.mat"
        scipy.io.savemat(metrics_path, {"accuracy": np.array(val)})
    # Line(f"Done simulation {args}")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
    Line(f"{args} done.")
    ray.shutdown()
    gc.collect()