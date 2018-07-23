ssh meta "cd HpBandSter/hpbandster/examples/example_branin/$1 && tar -cvzf result.tar.gz results.$2*"
scp meta:HpBandSter/hpbandster/examples/example_branin/$1/result.tar.gz $1
tar -xvzf $1/result.tar.gz -C $1

python3 plot_trajectories.py --run-filter $2 --working-dir $1