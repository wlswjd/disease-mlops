from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'superjinjungman',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='disease_model_training',
    default_args=default_args,
    start_date=datetime(2025, 6, 1),
    schedule_interval='@daily',  # 매일 1회 (또는 None 으로 수동 실행)
    catchup=False,
) as dag:

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /app/scripts/train.py',
    )

    train_model

