from flwr.common.logger import log

from logging import WARNING
from typing import Optional, Callable, Tuple, Dict, List, Union

from common.typing import (
    Weights,
    Scalar,
    Parameters, 
    FitIns,
    FitRes,
    MetricsAggregationFn,
    EvaluateIns,
    EvaluateRes,
)
from common.parameter import parameters_to_weights, weights_to_parameters
from hfl_server_app.strategy.strategy import HFLStrategy
from hfl_server_app.fog_manager import FogServerManager
from hfl_server_app.fog_proxy import FogServerProxy

from .aggregate import aggregate, weighted_loss_avg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""

class FedAvg(HFLStrategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_eval: float = 1.0,
        min_fit_fogs: int = 2,
        min_eval_fogs: int = 2,
        min_available_fogs: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        ) -> None:
        super().__init__()
        
        if (
            min_fit_fogs > min_available_fogs
            or min_eval_fogs > min_available_fogs
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_fogs = min_fit_fogs
        self.min_eval_fogs = min_eval_fogs
        self.min_available_fogs = min_available_fogs
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedFog(accecpt_failures={self.accept_failures})"
        return rep

    def num_fit_fogs(
        self,
        num_available_fogs: int)->Tuple[int, int]:
        num_fogs = int(num_available_fogs * self.fraction_fit)
        return max(num_fogs, self.min_fit_fogs), self.min_available_fogs

    def num_evaluation_fogs(
        self,
        num_available_fogs: int)->Tuple[int, int]:
        num_fogs = int(num_available_fogs * self.fraction_eval)
        return max(num_fogs, self.min_eval_fogs), self.min_available_fogs

    def initialize_parameters(
        self,
        fogserver_manager: FogServerManager) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
        self,
        rnd: int,
        parameters: Parameters,
        fogserver_manager: FogServerManager) -> List[Tuple[FogServerProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_fogs = self.num_fit_fogs(
            fogserver_manager.num_available()
        )
        fogs = fogserver_manager.sample(
            num_fogs=sample_size,
            min_num_fogs=min_num_fogs
        )
        return [(fog, fit_ins) for fog in fogs]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[FogServerProxy, FitRes]],
        failures: List[BaseException]) -> Union[Tuple[Optional[Parameters], Dict[str, Scalar]], Optional[Weights]]:
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}
        
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = weights_to_parameters(aggregate(weights_results))
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        return parameters_aggregated, metrics_aggregated
    
    def configure_evaluate(
        self,
        rnd: int,
        parameters: Parameters,
        fogserver_manager: FogServerManager) -> List[Tuple[FogServerProxy, EvaluateIns]]:
        # if self.eval_fn is not None:
        #     return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        if rnd >= 0:
            sample_size, min_num_fogs = self.num_evaluation_fogs(
                fogserver_manager.num_available()
            )
            fogs = fogserver_manager.sample(
                num_fogs=sample_size, min_num_fogs=min_num_fogs
            )
        else:
            fogs = list(fogserver_manager.all().values())

        return [(fog, evaluate_ins) for fog in fogs]

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[FogServerProxy, EvaluateRes]],
        failures: List[BaseException]) -> Union[Tuple[Optional[float], Dict[str, Scalar]], Optional[float]]:
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        # loss_aggregated = weighted_loss_avg(
        #     [
        #         (
        #             evaluate_res.num_examples,
        #             evaluate_res.loss,
        #             evaluate_res.accuracy,
        #         )
        #         for _, evaluate_res in results
        #     ]
        # )
        metrics_aggregated = {}
        for _, evaluate_res in results:
            for key, val in evaluate_res.metrics.items():
                metrics_aggregated[key] = val
        return _, metrics_aggregated

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics