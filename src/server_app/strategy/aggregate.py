"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Optional, Tuple

import numpy as np

from common.typing import Weights


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * (num_examples / num_examples_total) for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def cv_aggregate(cv_weights: Weights, results: List[Tuple[Weights, int]], num_clients: int) -> Weights:
    """Compute control variate average."""
    # Calculate the total number of examples used during training

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer / num_clients for layer in weights] for weights, _ in results
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]

    cv_aggregated = [val + prime for val, prime in zip(cv_weights, weights_prime)]

    return cv_aggregated

def weighted_loss_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    weights: Weights, deltas: List[Weights], hs_fll: List[Weights]
) -> Weights:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_weights = [(u - v) * 1.0 for u, v in zip(weights, updates)]
    return new_weights