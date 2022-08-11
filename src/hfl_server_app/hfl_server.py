"Flower hfl server"


import concurrent.futures
import time
import timeit
from logging import DEBUG, INFO, WARNING
from typing import Optional, Tuple, Union, List, Dict

from flwr.common.logger import log

from common.typing import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Disconnect,
    Reconnect,
    Scalar,
    Weights,
)
from common.parameter import weights_to_parameters
from server_app.history import History
from .fog_manager import FogServerManager
from .fog_proxy import FogServerProxy
from .strategy.strategy import HFLStrategy
from .strategy.fedavg import FedAvg

DEPRECATION_WARNING_EVALUATE = """
DEPRECATION WARNING: Method
    Server.evaluate(self, rnd: int) -> Optional[
        Tuple[Optional[float], EvaluateResultsAndFailures]
    ]
is deprecated and will be removed in a future release, use
    Server.evaluate_round(self, rnd: int) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]
instead.
"""

DEPRECATION_WARNING_EVALUATE_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_evaluate
return format:
    Strategy.aggregate_evaluate(...) -> Optional[float]
This format is deprecated and will be removed in a future release. It should use
    Strategy.aggregate_evaluate(...) -> Tuple[Optional[float], Dict[str, Scalar]]
instead.
"""

DEPRECATION_WARNING_FIT_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_fit
return format:
    Strategy.aggregate_fit(...) -> Optional[Weights]
This format is deprecated and will be removed in a future release. It should use
    Strategy.aggregate_fit(...) -> Tuple[Optional[Weights], Dict[str, Scalar]]
instead.
"""

FitResultsAndFailures = Tuple[List[Tuple[FogServerProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[FogServerProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[FogServerProxy, Disconnect]], List[BaseException]
]

class HFLServer:
    def __init__(self, fogserver_manager: FogServerManager, strategy: Optional[HFLStrategy] = None,) -> None:
        self._fogserver_manager = fogserver_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: HFLStrategy = strategy if strategy is not None else FedAvg()
    
    def set_strategy(self, strategy: HFLStrategy)->None:
        self.strategy = strategy

    def fogserver_manager(self)-> FogServerManager:
        return self._fogserver_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int)->History:
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])
        
        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(rnd=current_round)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            if current_round % 1 == 0:
                res_cen = self.strategy.evaluate(parameters=self.parameters)
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )
                    history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                    history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)
            if current_round % 1 == 0:
                # Evaluate model on a sample of available clients
                res_fed = self.evaluate_round(rnd=-1)
                if res_fed:
                    _, evaluate_metrics_fed, _ = res_fed
                    if evaluate_metrics_fed:
                        history.add_metrics_distributed(
                            rnd = current_round, metrics=evaluate_metrics_fed
                        )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history, self.parameters

    def fit_round(
        self, rnd: int
    )-> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        fog_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, fogserver_manager=self._fogserver_manager
        )

        if not fog_instructions:
            log(INFO, "fit_round: no fogservers selected, cancel")
            return None
        
        log(
            DEBUG,
            "fit_round: strategy sampled %s fogservers (out of %s)",
            len(fog_instructions),
            self._fogserver_manager.num_available(),
        )

        results, failures = fit_fogs(fog_instructions)
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[Weights],  # Deprecated
        ] = self.strategy.aggregate_fit(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = None
        elif isinstance(aggregated_result, list):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = weights_to_parameters(aggregated_result)
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)

    
    def evaluate(
        self,
        rnd: int)-> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        log(WARNING, DEPRECATION_WARNING_EVALUATE)
        res = self.evaluate_round(rnd)
        if res is None:
            return None
        # Deconstruct
        loss, _, results_and_failures = res
        return loss, results_and_failures
    
    def evaluate_round(
        self,
        rnd: int)-> Optional[Tuple[Optional[float], Dict[str, List[Tuple[int, Scalar]]], EvaluateResultsAndFailures]]:
        fog_instructions = self.strategy.configure_evaluate(
            rnd=rnd,
            parameters=self.parameters,
            fogserver_manager=self._fogserver_manager
        )

        if not fog_instructions:
            log(INFO, "evaluate_round: no fogservers selected, cancel")
            return None
        
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s fogservers (out of %s)",
            len(fog_instructions),
            self._fogserver_manager.num_available(),
        )

        results, failures = evaluate_fogs(fog_instructions)
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        aggregated_result: Union[
            Tuple[Optional[float], Dict[str, Scalar]],
            Optional[float],
        ] = self.strategy.aggregate_evaluate(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = None
        elif isinstance(aggregated_result, float):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = aggregated_result
        else:
            loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_fogs(self)-> None:
        all_fogs = self._fogserver_manager.all()
        _ = shutdown(fogs=[all_fogs[k] for k in all_fogs.keys()])

    def _get_initial_parameters(self)->Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            fogserver_manager=self._fogserver_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random fogserver")
        random_fog = self._fogserver_manager.sample(1)[0]
        parameters_res = random_fog.get_parameters()
        log(INFO, "Received initial parameters from one random fogserver")
        return parameters_res.parameters

def shutdown(fogs: List[FogServerProxy])->ReconnectResultsAndFailures:
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(reconnect_fog, fog, reconnect) for fog in fogs
        ]
        concurrent.futures.wait(futures)

    results: List[Tuple[FogServerProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_fog(
    fog: FogServerProxy,
    reconnect: Reconnect)->Tuple[FogServerProxy, Disconnect]:
    disconnect = fog.reconnect(reconnect)
    return fog, disconnect

def fit_fogs(
    fog_instructions: List[Tuple[FogServerProxy, FitIns]])->FitResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_fog, fog, ins) for fog, ins in fog_instructions
        ]
        concurrent.futures.wait(futures)

    results: List[Tuple[FogServerProxy, FitRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def fit_fog(
    fog: FogServerProxy,
    ins: FitIns)->Tuple[FogServerProxy, FitRes]:
    fit_res = fog.fit(ins)
    return fog, fit_res

def evaluate_fogs(
    fog_instructions: List[Tuple[FogServerProxy, EvaluateIns]])->EvaluateResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_fog, fog, ins) for fog, ins in fog_instructions
        ]
        concurrent.futures.wait(futures)

    results: List[Tuple[FogServerProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def evaluate_fog(
    fog: FogServerProxy,
    ins: EvaluateIns)->Tuple[FogServerProxy, EvaluateRes]:
    evaluate_res = fog.evaluate(ins)
    return fog, evaluate_res