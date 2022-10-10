from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.task_group import TaskGroup

pathScript = "~/ml-model-api/scripts/"

default_args = {
	"owner": "Kraisfeld",
	"depends_on_past":False,
	"start_date": datetime(2019,1,1),
	"retries":0
}

with DAG(
	"iris_pipeline",
	schedule_interval=None,
	catchup=False,
	default_args=default_args
	) as dag:

	start = DummyOperator(task_id="start")

	extract = BashOperator(
			dag = dag,
			task_id="extract",
			bash_command="""
			python {0}/extract.py
			""".format(pathScript)
		)

	with TaskGroup("train_model", tooltip="train_model") as etl:

		t1 = BashOperator(
				dag = dag,
				task_id="train_model_1",
				bash_command="""
				cd {0}
				python model_1.py
				""".format(pathScript)
			)

		t2 = BashOperator(
				dag = dag,
				task_id="train_model_2",
				bash_command="""
				cd {0}
				python model_2.py
				""".format(pathScript)
			)

		t3 = BashOperator(
				dag = dag,
				task_id="find_best_model",
				bash_command="""
				cd {0}
				python find_best_model.py
				""".format(pathScript)
			)

		[[t1,t2] >> t3]

	end = DummyOperator(task_id="end")

	start >> extract >> etl >> end
