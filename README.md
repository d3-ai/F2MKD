# Usage
## Setup
See the `Pipfile` for python environment requirements.
```
pipenv update
```

## Dataset partitioning
The dataset partitioning is inspired by [Federated Learning on Non-IID Data Silos: An Experimental Study.](https://arxiv.org/pdf/2102.02079.pdf)

```
. ./shell/fog_partitions.sh {fog_partitions} {client_partitions}
. ./shell/client_partitions.sh {fog_partitions} {client_partitions}
```
* `fog_partitions`: dataset partitioning for fog servers 
* `client_partitions`: dataset partitioning for clients

There are three variants for partitioning.
* `iid`: each participant has 10 labels
* `noniid-label2`: each participant has 2 labels
* `noniid-label1`: each participant has 1 label

## Run Federated Learning
To simulate the large cohorts of clients, we leverage `Ray` following [the code examples provided by Flower.](https://github.com/adap/flower/tree/main/examples/simulation_pytorch)
```
. ./run_fedfog.sh {fog_partitions} {client_partitions}
. ./run_fml.sh {fog_partitions} {client_partitions} {alpha} {beta}
. ./run_f2mkd.sh {fog_partitions} {client_partitions} {alpha} {beta}
```
`alpha` and `beta` are FL strategy-specific hyperparameters.