import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import logging
import matplotlib.pyplot as plt
import math
import random
import itertools

from hpbandster.optimizers import BOHB, HyperBand, RandomSearch

import util
from worker import BraninWorker

logging.basicConfig(level=logging.INFO)

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x1', lower=-5, upper=10))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x2', lower=0, upper=15))


# Every run has to have a unique (at runtime) id.
# This needs to be unique for concurent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
# Here we pick '0'
run_id = '0'

# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine
# with a random port
NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

connection = dict(
    nameserver=ns_host,
    nameserver_port=ns_port,
    run_id=run_id,  # unique Hyperband run id
)

# Start a bunch of workers in some threads, just to show how it works.
# On the cluster, each worker would run in a separate job and the nameserver
# credentials have to be distributed.
num_workers = 1

x1 = random.uniform(-5, 10)
x2 = random.uniform(0, 15)
y = BraninWorker.calc_branin(x1, x2)

config = dict(
    configspace=config_space,
    min_budget=1,
    max_budget=100,
)

# As baseline run the different algorithms
for constructor in [HyperBand]:
    util.start_worker(
        num_workers,
        {**connection, 'true_y': y}
    )

    res = util.run_hb(
        constructor,
        {**connection, **config, 'ping_interval': 3600},
        num_workers)

    incumbent_trajectory = util.log_results(res, simulate_time=True)

    times = incumbent_trajectory['times_finished']
    losses = incumbent_trajectory['losses']

    plt.plot(times, losses, label=str(constructor))


def cost1(z1: float, z2: float, z3: float) -> float:
    return 0.05 + (z1**3 * 1**2 * 1**1.5)


def cost2(z1: float, z2: float, z3: float) -> float:
    return 0.05 + (1**3 * z2**2 * 1**1.5)


def cost3(z1: float, z2: float, z3: float) -> float:
    return 0.05 + (1**3 * 1**2 * z3**1.5)

# Test differentcost functions
for cost_fun in [cost1, cost2, cost3]:
    util.start_worker(
        num_workers,
        {**connection, 'true_y': y, 'cost': cost_fun}
    )

    res = util.run_hb(
        HyperBand,
        {**connection, **config, 'ping_interval': 3600},
        num_workers)

    incumbent_trajectory = util.log_results(res, simulate_time=True)

    times = incumbent_trajectory['times_finished']
    losses = incumbent_trajectory['losses']

    plt.plot(times, losses, label=str(cost_fun))


print("x1: {}, x2: {}, y: {}".format(x1, x2, y))

plt.xlabel('wall clock time [s]')
plt.ylabel('incumbent loss')
plt.legend()
plt.show()
