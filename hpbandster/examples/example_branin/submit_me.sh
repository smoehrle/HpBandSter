# submit via qsub -t 1-4 submit_me.sh

#$ -q test_core.q
#$ -cwd
#$ -o ~moehrles/logs/$JOB_ID/$TASK_ID.o
#$ -e ~moehrles/logs/$JOB_ID/$TASK_ID.e

# create logs directory if none exists
mkdir -p "~moehrles/logs/${$JOB_ID}"

# enter the virtual environment
source ~moehrles/HpBandSter/venv/bin/activate

python3 run_me_cluster.py --run_id "$JOB_ID" --last-task-id "$LAST_TASK_ID" --task_id "$TASK_ID" --master --worker --config $1
