# submit via qsub -t 1-4 submit_me.sh

#$ -cwd
#$ -o $JOB_ID.$TASK_ID.o
#$ -e $JOB_ID.$TASK_ID.e

# enter the virtual environment
source ~moehrles/HpBandSter/venv/bin/activate

python3 run_me_cluster.py --run_id "$JOB_ID" --last-task-id "$SGE_LAST_TASK_ID" --task_id "$SGE_TASK_ID" --master --worker --config runs/ex1/config.yml
