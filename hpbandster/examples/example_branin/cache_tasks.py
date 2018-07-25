import openml
import pandas as pd

tag = "study_99"

print("Download task list for tag {} ...".format(tag))
tasks_dict = openml.tasks.list_tasks(tag=tag)
tasks_df = pd.DataFrame.from_dict(tasks_dict, orient='index')
tids = tasks_df.tid.values
print("")
for tid in tids:
    print("Download task and dataset {} ...".format(tid))
    task = openml.tasks.get_task(tid)
    task.get_dataset()
