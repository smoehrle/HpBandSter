# submit via qsub -t 1-4 submit_me.sh

#$ -q test_core.q
#$ -cwd
#$ -o ~kuenstld/logs/$JOB_ID/$TASK_ID.o
#$ -e ~kuenstld/logs/$JOB_ID/$TASK_ID.e

# create logs directory if none exists
mkdir -p "~kuenstld/logs/${$JOB_ID}"

# enter the virtual environment
source ~kuenstld/virtualenvs/HpBandSter/bin/activate

if [[ "$SGE_TASK_ID" == 1 ]] ; then
    MASTER='--master'
fi

python3 run_me_cluster.py --run_id "$JOB_ID" "$MASTER" --worker --working_dir .
