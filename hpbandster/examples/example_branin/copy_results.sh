ssh meta "cd HpBandSter/hpbandster/examples/example_branin/$1 && tar -cvzf result.tar.gz results.$2*"
scp meta:HpBandSter/hpbandster/examples/example_branin/$1/result.tar.gz $1
ssh meta "rm HpBandSter/hpbandster/examples/example_branin/$1/result.tar.gz"
tar -xvzf $1/result.tar.gz -C $1
rm $1/result.tar.gz

python3 plot_trajectories.py --run-filter $2 --working-dir $1