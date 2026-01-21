import os
from comet_ml import Experiment
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

        # Initialize Comet Experiment
        self.experiment = Experiment(
            project_name="cometml-project",
            api_key="dGf92K7ftZtPr2brhige42CLq",
            workspace="shahidsak1973",
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_output_logging="simple"
        )

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data split successfully")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data", e)

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing LGBM model")

            lgbm_model = lgb.LGBMClassifier(
                random_state=self.random_search_params["random_state"]
            )

            logger.info("Starting Hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )

            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            best_params = random_search.best_params_

            logger.info(f"Best parameters: {best_params}")

            # Log parameters to Comet
            self.experiment.log_parameters(best_params)

            return best_model

        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model", e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model")

            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }

            for key, value in metrics.items():
                logger.info(f"{key.upper()} : {value}")

            # Log metrics to Comet
            self.experiment.log_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model", e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving model")
            joblib.dump(model, self.model_output_path)

            # Log model artifact to Comet
            self.experiment.log_model(
                name="lgbm_booking_model",
                file_or_folder=self.model_output_path
            )

            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model", e)

    def run(self):
        try:
            logger.info("Starting Model Training pipeline")

            # Log datasets
            self.experiment.log_asset(self.train_path)
            self.experiment.log_asset(self.test_path)

            X_train, y_train, X_test, y_test = self.load_and_split_data()
            best_model = self.train_lgbm(X_train, y_train)
            self.evaluate_model(best_model, X_test, y_test)
            self.save_model(best_model)

            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed {e}")
            raise CustomException("Failed during model training pipeline", e)

        finally:
            self.experiment.end()


if __name__ == "__main__":
    trainer = ModelTraining(
        PROCESSED_TRAIN_DATA_PATH,
        PROCESSED_TEST_DATA_PATH,
        MODEL_OUTPUT_PATH
    )
    trainer.run()
