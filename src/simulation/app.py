"""Flower simulation app."""
from logging import INFO
from typing import Any, Callable, Dict, List, Optional

import ray

from flwr.common.logger import log
from client_app.dml_client import DMLClientProxy

from server_app.history import History
from server_app.app import _fl
from server_app.client_manager import SimpleClientManager
from server_app.server import Server
from server_app.strategy.strategy import Strategy
from client_app.client import Client
from .ray_transport.ray_client_proxy import RayClientProxy
from .ray_transport.ray_dml_client_proxy import RayDMLClientProxy

INVALID_ARGUMENTS_START_SIMULATION = """
INVALID ARGUMENTS ERROR
Invalid Arguments in method:
`start_simulation(
    *,
    client_fn: Callable[[str], Client],
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, int]] = None,
    num_rounds: int = 1,
    strategy: Optional[Strategy] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> None:`
REASON:
    Method requires:
        - Either `num_clients`[int] or `clients_ids`[List[str]]
        to be set exclusively.
        OR
        - `len(clients_ids)` == `num_clients`
"""

def start_simulation(  # pylint: disable=too-many-arguments
    *,
    strategy: Optional[Strategy],
    num_clients: Optional[int],
    num_rounds: int,
    client_fn: Callable[[str], Client],
    client_config: Optional[Dict[str, Any]],
    client_resources: Optional[Dict[str, int]] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,) -> History:
    # clients_ids takes precedence
    cids: List[str] = [str(x) for x in range(num_clients)]

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
    ray.init(**ray_init_args)
    log(INFO,"Ray initialized with resources: %s",ray.cluster_resources(),)

    # Initialize server and server config
    config = {"num_rounds": num_rounds}
    client_manager = SimpleClientManager()
    initialized_server = Server(client_manager=client_manager,strategy=strategy)
    log(INFO,"Starting Flower simulation running: %s",config,)

    # Register one RayClientProxy object for each client with the ClientManager
    resources = client_resources if client_resources is not None else {}
    for cid in cids:
        client_proxy = RayClientProxy(
            client_fn=client_fn,
            cid=cid,
            resources=resources,
            config=client_config,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    hist, params = _fl(
        server=initialized_server,
        config=config,
        force_final_distributed_eval=False,
    )
    return hist, params


def start_dml_simulation(  # pylint: disable=too-many-arguments
    *,
    strategy: Optional[Strategy],
    num_clients: Optional[int],
    num_rounds: int,
    client_config: Optional[Dict[str, Any]],
    client_resources: Optional[Dict[str, int]] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,) -> History:
    # clients_ids takes precedence
    cids: List[str] = [str(x) for x in range(num_clients)]

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
    ray.init(**ray_init_args)
    log(INFO,"Ray initialized with resources: %s",ray.cluster_resources(),)

    # Initialize server and server config
    config = {"num_rounds": num_rounds}
    client_manager = SimpleClientManager()
    initialized_server = Server(client_manager=client_manager,strategy=strategy)
    log(INFO,"Starting Flower simulation running: %s",config,)

    # Register one RayClientProxy object for each client with the ClientManager
    resources = client_resources if client_resources is not None else {}
    for cid in cids:
        client_proxy = DMLClientProxy(
            cid=cid,
            initial_parameters=strategy.initial_parameters,
            config=client_config,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    hist, params = _fl(
        server=initialized_server,
        config=config,
        force_final_distributed_eval=False,
    )
    return hist, params

