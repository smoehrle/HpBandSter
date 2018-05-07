import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
import logging
import matplotlib.pyplot as plt
import math
import random
import itertools


from hpbandster.optimizers import BOHB, HyperBand, RandomSearch

from worker import MyWorker

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

# Start a bunch of workers in some threads, just to show how it works.
# On the cluster, each worker would run in a separate job and the nameserver
# credentials have to be distributed.
num_workers = 1

x1 = random.uniform(-5, 10)
x2 = random.uniform(0, 15)
y = MyWorker.calc_branin(x1, x2)

workers = []
for i in range(num_workers):
    w = MyWorker(
        true_y=y,
        nameserver=ns_host,
        nameserver_port=ns_port,
        run_id=run_id,  # unique Hyperband run id
        id=i  # unique ID as all workers belong to the same process
    )
    w.run(background=True)
    workers.append(w)


for constructor in [RandomSearch, HyperBand, BOHB]:
    print('Start {} run'.format(constructor))
    HB = constructor(
        run_id=run_id,
        configspace=config_space,
        min_budget=9,
        max_budget=243,
        nameserver=ns_host,
        nameserver_port=ns_port,
        ping_interval=3600
    )

    res = HB.run(3, min_n_workers=num_workers)
    HB.shutdown()

    id2config = res.get_id2config_mapping()

    print('A total of {} unique configurations where sampled.'.format(
        len(id2config.keys())))
    runs = sorted(res.get_all_runs(), key=lambda x: x.budget)
    print('A total of {} runs where executed.'.format(len(runs)))
    for b, r in itertools.groupby(runs, key=lambda x: x.budget):
        print('Budget: {:3}, #Runs: {:3}'.format(b, sum([1 for _ in r])))
    sum_ = sum([x.budget for x in runs])
    print('Total budget: {}'.format(sum_))
    for k in id2config.items():
        print("Key: {}".format(k))

    incumbent_trajectory = res.get_incumbent_trajectory()

    ids = incumbent_trajectory['config_ids']
    times = incumbent_trajectory['times_finished']
    budgets = incumbent_trajectory['budgets']
    losses = incumbent_trajectory['losses']

    for i in range(len(ids)):
        print("Id: ({0[0]}, {0[1]}, {0[2]:>2}), time: {1:>5.2f}, budget: {2:>5}, loss: {3:>5.2f}".format(ids[i], times[i], budgets[i], losses[i]))

    plt.plot(
        incumbent_trajectory['times_finished'],
        incumbent_trajectory['losses'],
        label=str(constructor)
    )

print("x1: {}, x2: {}, y: {}".format(x1, x2, y))

plt.xlabel('wall clock time [s]')
plt.ylabel('incumbent loss')
plt.legend()
plt.show()

HB.shutdown(shutdown_workers=True)
