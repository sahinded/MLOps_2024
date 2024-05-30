import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Build or Connect mlflow experiment
    EXPERIMENT_NAME = "random-forest-train"
    mlflow.set_experiment(EXPERIMENT_NAME)
            
    # before your training code to enable automatic logging of sklearn metrics, params, and models
    mlflow.sklearn.autolog()    
    
    with mlflow.start_run():
        # Optional: Set some information about Model
    
        mlflow.set_tag("algorithm", "Machine Learning")
        mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
        mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
        mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')        
        
        # Set Model params information
        params = {"max_depth": 10, "random_state": 0}
        mlflow.log_params(params)



        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        
        val_rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("val_rmse", val_rmse)

        # Log Model two options
        # Option1: Just only model in log
        mlflow.sklearn.log_model(sk_model = rf, artifact_path = "models_mlflow")
                
        # Option 2: save Model, and Optional: Preprocessor or Pipeline in log
        # Create dest_path folder unless it already exists
        # pathlib.Path(dest_path).mkdir(exist_ok=True)
        os.makedirs('model', exist_ok=True)
        pickle_path = os.path.join('model', "rf_model.pkl")
        with open(pickle_path, 'wb') as f_out:
            pickle.dump(rf, f_out)
            
        # whole proccess like pickle, saved Model, Optional: Preprocessor or Pipeline
        mlflow.log_artifact(local_path = pickle_path, artifact_path="models_pickle")        
        
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

if __name__ == '__main__':
    run_train()
