"""Flower client (abstract base class)."""


from abc import ABC, abstractmethod

from common.typing import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    Properties,
    PropertiesIns,
    PropertiesRes,
    Reconnect,
)


class ClientProxy(ABC):
    """Abstract base class for Flower client proxies."""

    def __init__(self, cid: str):
        self.cid = cid
        self.properties: Properties = {}

    @abstractmethod
    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""

    @abstractmethod
    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        """Returns the client's properties."""

    @abstractmethod
    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""

    @abstractmethod
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

    @abstractmethod
    def reconnect(self, reconnect: Reconnect) -> Disconnect:
        """Disconnect and (optionally) reconnect later."""