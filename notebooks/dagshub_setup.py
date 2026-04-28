import mlflow

import dagshub

mlflow.set_tracking_uri('https://dagshub.com/shivamtiwari83032-collab/mini-mlops-proj.mlflow')
dagshub.init(repo_owner='shivamtiwari83032-collab', repo_name='mini-mlops-proj', mlflow=True)
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)



  # 27ac23ebe84b70294deff5c3a18f9fa1f97a03d8