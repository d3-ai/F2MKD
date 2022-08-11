"""Flower FogServer strategy."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Dict

from common.typing import (
    Parameters,
    FitIns,
    FitRes,
    Scalar,
    Weights,
)
from server_app.client_manager import ClientManager
from server_app.client_proxy import ClientProxy

class FogServerStrategy(ABC):
    """
    Abstract base class for fogserver strategy in hierarchical setting.
    The strategy is the same with the Strategy Flower provided.
    """

    @abstractmethod
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.
        Parameters
        ----------
            client_manager: ClientManager. The client manager which holds all currently
                connected clients.
        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the fogserver will treat these as the
            initial global model parameters.
        """

    @abstractmethod
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        Returns
        -------
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
        is not included in this list, it means that this `ClientProxy`
        will not participate in the next round of federated learning.
        """

    @abstractmethod
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Union[
        Tuple[Optional[Parameters], Dict[str, Scalar]],
        Optional[Weights],  # Deprecated
    ]:
        """Aggregate training results.
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[BaseException]
            Exceptions that occurred while the fogserver was waiting for client
            updates.
        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the fogserver will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the fogserver will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
    