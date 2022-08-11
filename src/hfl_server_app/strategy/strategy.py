"""Flower hfl server strategy."""


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from hfl_server_app.fog_manager import FogServerManager
from hfl_server_app.fog_proxy import FogServerProxy


class HFLStrategy(ABC):
    """Abstract base class for server strategy in hierarchical settings."""
    @abstractmethod
    def initialize_parameters(
        self,
        fogserver_manager: FogServerManager)-> Optional[Parameters]:
        """
        Initialize the (global) model parameters.
        Parameters
        ----------
            fogserver_manager: FogServerManager. The fogserver manager which holds all currently
                connected fogservers.
        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """

    @abstractmethod
    def configure_fit(
        self,
        rnd: int,
        parameters: Parameters,
        fogserver_manager: FogServerManager)-> List[Tuple[FogServerProxy, FitIns]]:
        """Configure the next round of training.
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        fogserver_manager : FogServerManager
            The client manager which holds all currently connected fogservers.
        Returns
        -------
        A list of tuples. Each tuple in the list identifies a `FogServerProxy` and the
        `FitIns` for this particular `FogServerProxy`. If a particular `FogServerProxy`
        is not included in this list, it means that this `FogServerProxy`
        will not participate in the next round of federated learning.
        """
    
    @abstractmethod
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[FogServerProxy, FitRes]],
        failures: List[BaseException])->Union[Tuple[Optional[Parameters], Dict[str, Scalar]], Optional[Weights]]:
        """Aggregate training results.
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        results : List[Tuple[FogServerProxy, FitRes]]
            Successful updates from the previously selected and configured
            fogservers. Each pair of `(FogServerProxy, FitRes)` constitutes a
            successful update from one of the previously selected fogservers. Not
            that not all previously selected fogservers are necessarily included in
            this list: a fogserver might drop out and not submit a result. For each
            fogserver that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[BaseException]
            Exceptions that occurred while the server was waiting for fogserver
            updates.
        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """

    @abstractmethod
    def configure_evaluate(
        self,
        rnd: int,
        parameters: Parameters,
        fogserver_manager: FogServerManager)->List[Tuple[FogServerProxy, EvaluateIns]]:
        """Configure the next round of evaluation.
        Arguments:
            rnd: Integer. The current round of federated learning.
            parameters: Parameters. The current (global) model parameters.
            fogserver_manager: FogServerManager. The fogserver manager which holds all currently
                connected fogservers.
        Returns:
            A list of tuples. Each tuple in the list identifies a `FogServerProxy` and the
            `EvaluateIns` for this particular `FogServerProxy`. If a particular
            `FogServerProxy` is not included in this list, it means that this
            `FogServerProxy` will not participate in the next round of federated
            evaluation.
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[FogServerProxy, EvaluateRes]],
        failures: List[BaseException])->Union[Tuple[Optional[float], Dict[str, Scalar]], Optional[float]]:
        """Aggregate evaluation results.
        Arguments:
            rnd: int. The current round of federated learning.
            results: List[Tuple[FogServerProxy, FitRes]]. Successful updates from the
                previously selected and configured fogservers. Each pair of
                `(FogServerProxy, FitRes` constitutes a successful update from one of the
                previously selected fogservers. Not that not all previously selected
                fogservers are necessarily included in this list: a fogserver might drop out
                and not submit a result. For each fogserver that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: List[BaseException]. Exceptions that occurred while the server
                was waiting for fogserver updates.
        Returns:
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """

    @abstractmethod
    def evaluate(
        self,
        parameters: Parameters)-> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters.
        This function can be used to perform centralized (i.e., server-side) evaluation
        of model parameters.
        Arguments:
            parameters: Parameters. The current (global) model parameters.
        Returns:
            The evaluation result, usually a Tuple containing loss and a
            dictionary containing task-specific metrics (e.g., accuracy).
        """



