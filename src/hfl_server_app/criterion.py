"""Abstract class for criterion sampling."""


from abc import ABC, abstractmethod

from .fog_proxy import FogServerProxy


class Criterion(ABC):
    """Abstract class which allows subclasses to implement criterion
    sampling."""

    @abstractmethod
    def select(self, fog: FogServerProxy) -> bool:
        """Decide whether a fogserver should be eligible for sampling or not."""