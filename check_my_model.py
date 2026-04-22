import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/shivamtiwari83032-collab/mini-mlops-proj.mlflow")
client = MlflowClient()

experiment = client.get_experiment_by_name("LoR Hyperparameter Tuning")

print(f"Listing last 10 runs in: {experiment.name}\n")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=10)

for run in runs:
    name = run.data.tags.get('mlflow.runName', 'Unnamed')
    run_id = run.info.run_id
    # Check if this run has any files
    artifacts = client.list_artifacts(run_id)
    has_files = "YES ✅" if artifacts else "NO ❌"
    
    print(f"Run: {name}")
    print(f"ID:  {run_id}")
    print(f"Has Files? {has_files}")
    if artifacts:
        for a in artifacts:
            print(f"  --> Found: {a.path}")
    print("-" * 40)