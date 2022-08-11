"""Flower FogServer (abstract base class)"""

from abc import ABC, abstractmethod

from common.typing import (
    Properties,
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

class FogServerProxy(ABC):
    """Abstract base class for Flower fogserver proxies"""

    def __init__(self, fid: str):
        self.fid = fid
        self.properties: Properties = {}

    @abstractmethod
    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""

    @abstractmethod
    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        """Returns the fogserver's properties."""

    @abstractmethod
    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""

    @abstractmethod
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

    @abstractmethod
    def reconnect(self, reconnect: Reconnect) -> Disconnect:
        """Disconnect and (optionally) reconnect later."""