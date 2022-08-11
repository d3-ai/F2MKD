"""Ray-based Flower FogServerProxy implementation"""
import ray

from typing import Callable, Dict, Any, cast

from common.typing import (
    Disconnect,
    Reconnect,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
)
from fog_app.fog import FogServer

from hfl_server_app.fog_proxy import FogServerProxy

from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log
FogServerFn = Callable[[str, Dict[str, Any]], FogServerProxy]

class RayFogServerProxy(FogServerProxy):
    """Flower fogserver proxy which delegates work using Ray."""
    def __init__(self, fog_fn: FogServerFn, fid: str, resources: Dict[str, int], config: Dict[str, Any]):
        super().__init__(fid)
        self.fog_fn = fog_fn
        self.resources = resources
        self.config = config
    
    def get_parameters(self) -> ParametersRes:
        future_parameters_res = launch_and_get_parameters.options(
            **self.resources
        ).remote(self.fog_fn, self.fid, self.config)
        res = ray.worker.get(future_parameters_res)
        return cast(ParametersRes, res)

    def get_properties(self, ins: PropertiesIns)-> PropertiesRes:
        future_properties_res = launch_and_get_properties.options(
            **self.resources
        ).remote(self.fog_fn, self.fid, self.config, ins)
        res = ray.worker.get(future_properties_res)
        return cast(
            PropertiesRes,
            res,
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        future_fit_res = launch_and_fit.options(**self.resources).remote(
            self.fog_fn, self.fid, self.config, ins
        )
        res = ray.worker.get(future_fit_res)
        return cast(
            FitRes,
            res,
        )
    
    def evaluate(self, ins: EvaluateIns)-> EvaluateRes:
        future_evaluate_res = launch_and_evaluate.options(
            **self.resources
        ).remote(self.fog_fn, self.fid, self.config, ins)
        res = ray.worker.get(future_evaluate_res)
        return cast(
            EvaluateRes,
            res,
        )
    
    def reconnect(self, reconnect: Reconnect)->Disconnect:
        return Disconnect(reason="")
@ray.remote
def launch_and_get_parameters(
    fog_fn: FogServerFn,
    fid: str,
    config: Dict[str, Any])-> ParametersRes:
    fog: FogServer = _create_fog(fog_fn, fid, config)
    return fog.get_parameters()

@ray.remote
def launch_and_get_properties(
    fog_fn: FogServerFn,
    fid: str,
    config: Dict[str, Any],
    properties_ins: PropertiesIns)->PropertiesRes:
    fog: FogServer = _create_fog(fog_fn, fid, config)
    return fog.get_properties(properties_ins)

@ray.remote
def launch_and_fit(
    fog_fn: FogServerFn,
    fid: str,
    config: Dict[str, Any],
    fit_ins: FitIns)->FitRes:
    fog: FogServer = _create_fog(fog_fn, fid, config)
    return fog.fit(fit_ins)

@ray.remote
def launch_and_evaluate(
    fog_fn: FogServerFn,
    fid: str,
    config: Dict[str, Any],
    evaluate_ins: EvaluateIns)->EvaluateRes:
    fog: FogServer = _create_fog(fog_fn, fid, config)
    return fog.evaluate(evaluate_ins)

def _create_fog(
    fog_fn: FogServerFn,
    fid: str,
    config: Dict[str, Any])->FogServer:
    fog: FogServer = fog_fn(fid, config)
    return fog