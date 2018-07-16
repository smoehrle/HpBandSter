# submit via qsub -t 1-4 submit_me.sh

#$ -cwd
#$ -o logs/$JOB_ID.$TASK_ID.o
#$ -e logs/$JOB_ID.$TASK_ID.e

# enter the virtual environment
source $HOME/HpBandSter/venv/bin/activate

python3 run_me_cluster.py --run-id "$JOB_ID" --last-task-id "$SGE_TASK_LAST" --task-id "$SGE_TASK_ID" --master --worker --config $1
