"""Ray-based Flower ClientProxy implementation."""
from typing import Callable, Dict, Union, cast, Any

import ray

from client_app.client import Client
from server_app.client_proxy import ClientProxy

from common.typing import (
    Parameters,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Reconnect,
    Disconnect,
)

ClientFn = Callable[[str, Dict[str, Any]], Client]

class RayDMLClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""
    def __init__(self, client_fn: ClientFn, cid: str, init_params: Parameters, resources: Dict[str, int], config: Dict[str, Any]):
        super().__init__(cid)
        self.client_fn = client_fn
        self.resources = resources
        self.config = config
        self.parameters = init_params

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        future_paramseters_res = launch_and_get_parameters.options(
            **self.resources
        ).remote(self.client_fn, self.cid, self.config)
        res = ray.worker.get(future_paramseters_res)
        return cast(
            ParametersRes,
            res,
        )

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        """Returns client's properties."""
        future_properties_res = launch_and_get_properties.options(
            **self.resources
        ).remote(self.client_fn, self.cid, self.config, ins)
        res = ray.worker.get(future_properties_res)
        return cast(
            PropertiesRes,
            res,
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Train model parameters on the locally held dataset."""
        ins.dual_parameters = self.parameters
        future_fit_res = launch_and_fit.options(**self.resources).remote(
            self.client_fn, self.cid, self.config, ins
        )
        res = ray.worker.get(future_fit_res)
        fitres = cast(FitRes,res)
        self.parameters = fitres.dual_parameters
        fitres.dual_parameters = None
        return fitres

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        ins.dual_parameters = self.parameters
        # future_evaluate_res = launch_and_evaluate.options(**self.resources).remote(
        future_evaluate_res = launch_and_evaluate.remote(
            self.client_fn, self.cid, self.config, ins
        )
        res = ray.worker.get(future_evaluate_res)
        return cast(
            EvaluateRes,
            res,
        )

    def reconnect(self, reconnect: Reconnect) -> Disconnect:
        """Disconnect and (optionally) reconnect later."""
        return Disconnect(reason="")  # Nothing to do here (yet)



@ray.remote  # type: ignore
def launch_and_get_parameters(client_fn: ClientFn, cid: str, config: Dict[str, Any]) -> ParametersRes:
    """Exectue get_parameters remotely."""
    client: Client = _create_client(client_fn, cid, config)
    return client.get_parameters()


@ray.remote# (num_cpus=1,num_gpus=0.1)  # type: ignore
def launch_and_fit(client_fn: ClientFn, cid: str, config: Dict[str, Any], fit_ins: FitIns, )-> FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid, config)
    return client.fit(fit_ins)

@ray.remote  # type: ignore
def launch_and_get_properties(client_fn: ClientFn, cid: str, config: Dict[str, Any], properties_ins: PropertiesIns)-> PropertiesRes:
    """Exectue get_properties remotely."""
    client: Client = _create_client(client_fn, cid, config)
    return client.get_properties(properties_ins)


@ray.remote(num_cpus=1,num_gpus=0.05)  # type: ignore
def launch_and_evaluate(client_fn: ClientFn, cid: str, config: Dict[str, Any], evaluate_ins: EvaluateIns)-> EvaluateRes:
    """Exectue evaluate remotely."""
    client: Client = _create_client(client_fn, cid, config)
    return client.evaluate(evaluate_ins)


def _create_client(client_fn: ClientFn, cid: str, config: Dict[str, Any]) -> Client:
    """Create a client instance."""
    client: Client = client_fn(cid, config)
    return client
