"""Flower server app."""

from logging import INFO
from typing import Dict

from flwr.common.logger import log


from .hfl_server import HFLServer

DEFAULT_SERVER_ADDRESS = "[::]:8080"

def _hfl(
    server: HFLServer,
    config: Dict[str, int],
    force_final_distributed_eval: bool
) -> None:
    # Fit model
    hist, params = server.fit(num_rounds=config["num_rounds"])
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    if force_final_distributed_eval:
        # Temporary workaround to force distributed evaluation
        server.strategy.eval_fn = None  # type: ignore

        # Evaluate the final trained model
        res = server.evaluate_round(rnd=-1)
        if res is not None:
            loss, _, (results, failures) = res
            log(INFO, "app_evaluate: federated loss: %s", str(loss))
            log(
                INFO,
                "app_evaluate: results %s",
                str([(res[0].cid, res[1]) for res in results]),
            )
            log(INFO, "app_evaluate: failures %s", str(failures))
        else:
            log(INFO, "app_evaluate: no evaluation result")

    # Graceful shutdown
    server.disconnect_all_fogs()
    return hist, params