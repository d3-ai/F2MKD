"""Flower client (abstract base class)."""


from abc import ABC, abstractmethod

from common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
)


class Client(ABC):
    """Abstract base class for Flower clients."""

    @abstractmethod
    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters.
        Returns
        -------
        ParametersRes
            The current local model parameters.
        """

    @abstractmethod
    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        """Return set of client's properties.
        Returns
        -------
        PropertiesRes
            Client's properties.
        """

    @abstractmethod
    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset.
        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.
        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """

    @abstractmethod
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset.
        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.
        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """