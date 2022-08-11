"""Flower simulation app."""
from logging import INFO
from typing import Any, Callable, Dict, List, Optional

import ray

from flwr.common.logger import log
from fog_app.mkd_fogserver_proxy import MKDFogServerProxy
from fog_app.avg_fogserver_proxy import AvgFogServerProxy

from server_app.history import History
from hfl_server_app.app import _hfl
from hfl_server_app.hfl_server import HFLServer
from hfl_server_app.fog_manager import SimpleFogServerManager
from hfl_server_app.strategy.strategy import HFLStrategy

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
    strategy: Optional[HFLStrategy],
    num_fogs: Optional[int],
    num_rounds: int,
    fog_config: Optional[Dict[str, Any]],
    fog_resources: Optional[Dict[str, int]] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,) -> History:
    # fogsevers_ids takes precedence
    fids: List[str] = [str(x) for x in range(num_fogs)]

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
    fogserver_manager = SimpleFogServerManager()
    initialized_server = HFLServer(fogserver_manager=fogserver_manager, strategy=strategy)
    log(INFO,"Starting Flower simulation running: %s",config,)

    # Register one RayClientProxy object for each client with the ClientManager
    resources = fog_resources if fog_resources is not None else {}
    for fid in fids:
        fogserver_proxy = AvgFogServerProxy(
            fid=fid,
            config=fog_config,
        )
        initialized_server.fogserver_manager().register(fog=fogserver_proxy)

    # Start training
    hist, params = _hfl(
        server=initialized_server,
        config=config,
        force_final_distributed_eval=False,
    )
    return hist, params

def start_mkd_simulation(  # pylint: disable=too-many-arguments
    *,
    strategy: Optional[HFLStrategy],
    num_fogs: Optional[int],
    num_rounds: int,
    fog_config: Optional[Dict[str, Any]],
    fog_resources: Optional[Dict[str, int]] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,) -> History:
    # fogsevers_ids takes precedence
    fids: List[str] = [str(x) for x in range(num_fogs)]

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
    fogserver_manager = SimpleFogServerManager()
    initialized_server = HFLServer(fogserver_manager=fogserver_manager, strategy=strategy)
    log(INFO,"Starting Flower simulation running: %s",config,)

    # Register one RayClientProxy object for each client with the ClientManager
    resources = fog_resources if fog_resources is not None else {}
    for fid in fids:
        fogserver_proxy = MKDFogServerProxy(
            fid=fid,
            config=fog_config,
            initial_parameters=strategy.initial_parameters,
        )
        initialized_server.fogserver_manager().register(fog=fogserver_proxy)

    # Start training
    hist, params = _hfl(
        server=initialized_server,
        config=config,
        force_final_distributed_eval=False,
    )
    return hist, params



