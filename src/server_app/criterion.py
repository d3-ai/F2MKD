"""Abstract class for criterion sampling."""


from abc import ABC, abstractmethod

from .client_proxy import ClientProxy


class Criterion(ABC):
    """Abstract class which allows subclasses to implement criterion
    sampling."""

    @abstractmethod
    def select(self, client: ClientProxy) -> bool:
        """Decide whether a client should be eligible for sampling or not."""