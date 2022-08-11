"""Flower type definitions."""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

Weights = List[np.ndarray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]
MetricsAggregationFn = Callable[[List[Tuple[int, Metrics]]], Metrics]

Config = Dict[str, Scalar]
Properties = Dict[str, Scalar]

class Code(Enum):
    """Client status codes."""
    OK = 0
    GET_PARAMETERS_NOT_IMPLEMENTED = 1

@dataclass
class Status:
    """Client status."""
    code: Code
    message: str

@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str


@dataclass
class ParametersRes:
    """Response when asked to return parameters."""

    parameters: Parameters


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]
    dual_parameters: Parameters = None
    parameters_dict: Dict[str, Parameters] = None


@dataclass
class FitRes:
    """Fit response from a client."""

    parameters: Parameters
    num_examples: int
    dual_parameters: Parameters = None
    parameters_dict: Dict[str, Parameters] = None
    metrics: Optional[Metrics] = None


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]
    dual_parameters: Parameters = None
    parameters_dict: Dict[str, Parameters] = None


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    loss: float
    num_examples: int
    accuracy: Optional[float] = None  # Deprecated
    metrics: Optional[Metrics] = None


@dataclass
class PropertiesIns:
    """Properties requests for a client."""

    config: Config


@dataclass
class PropertiesRes:
    """Properties response from a client."""

    properties: Properties


@dataclass
class Reconnect:
    """Reconnect message from server to client."""

    seconds: Optional[int]


@dataclass
class Disconnect:
    """Disconnect message from client to server."""

    reason: str