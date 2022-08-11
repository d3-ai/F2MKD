"""Flower Fog Manager"""

import random
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .fog_proxy import FogServerProxy
from .criterion import Criterion

class FogServerManager(ABC):
    """Abstract base class for managing Flower fogserver."""

    @abstractmethod
    def num_available(self) -> int:
        """Return the number of available fogservers."""

    @abstractmethod
    def register(self, fog: FogServerProxy) -> bool:
        """Register Flower FogServerProxy instance.
        Returns:
            bool: Indicating if registration was successful
        """

    @abstractmethod
    def unregister(self, fog: FogServerProxy) -> None:
        """Unregister Flower FogServerProxy instance."""

    @abstractmethod
    def all(self) -> Dict[str, FogServerProxy]:
        """Return all available fogservers."""

    @abstractmethod
    def wait_for(self, num_fogs: int, timeout: int) -> bool:
        """Wait until at least `num_fogs` are available."""

    @abstractmethod
    def sample(
        self,
        num_fogs: int,
        min_num_fogs: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[FogServerProxy]:
        """Sample a number of Flower FogServerProxy instances."""

class SimpleFogServerManager(FogServerManager):
    """Provides a pool of available fogservers."""
    def __init__(self) -> None:
        self.fogs: Dict[str, FogServerProxy] = {}
        self._cv = threading.Condition()

    def __len__(self)->int:
        return len(self.fogs)
    
    def wait_for(self, num_fogs: int, timeout: int=86400) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.fogs) >= num_fogs, timeout=timeout
            )
    
    def num_available(self) -> int:
        return len(self)

    def register(self, fog: FogServerProxy) -> bool:
        if fog.fid in self.fogs:
            return False

        self.fogs[fog.fid] = fog
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, fog: FogServerProxy) -> None:
        if fog.fid in self.fogs:
            del self.fogs[fog.fid]

            with self._cv:
                self._cv.notify_all()
    
    def all(self) -> Dict[str, FogServerProxy]:
        return self.fogs

    def sample(
        self,
        num_fogs: int,
        min_num_fogs: Optional[int] = None,
        criterion: Optional[Criterion] = None) -> List[FogServerProxy]:
        if min_num_fogs is None:
            min_num_fogs = num_fogs

        self.wait_for(min_num_fogs)
        available_fids = list(self.fogs)
        if criterion is not None:
            available_fids = [
                fid for fid in available_fids if criterion.select(self.fogs[fid])
            ]
        sampled_fids = random.sample(available_fids, num_fogs)
        return [self.fogs[fid] for fid in sampled_fids]