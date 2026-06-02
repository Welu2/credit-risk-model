import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "processed",
    "processed_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

target = "is_high_risk"

X = df.drop(columns=[target])
y = df[target]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MLFLOW SETUP
# =====================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Credit Risk Modeling"
)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    report = classification_report(y_test, y_pred)

    return metrics, report


# =====================================================
# MODEL CONFIGURATIONS
# =====================================================

models = {
    "LogisticRegression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    },

    "RandomForest": {
        "model": RandomForestClassifier(
            random_state=42
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10]
        }
    }
}

# =====================================================
# TRAINING LOOP
# =====================================================

best_model = None
best_f1 = -1
best_model_name = None
best_run_id = None

for model_name, config in models.items():

    with mlflow.start_run(run_name=model_name) as run:

        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,
            scoring="f1",
            n_jobs=-1
        )

        grid.fit(
            X_train,
            y_train
        )

        model = grid.best_estimator_

        metrics, report = evaluate_model(model)

        # -------------------------------------
        # Log parameters
        # -------------------------------------

        mlflow.log_params(
            grid.best_params_
        )

        # -------------------------------------
        # Log metrics
        # -------------------------------------

        mlflow.log_metrics(
            metrics
        )

        # -------------------------------------
        # Save classification report
        # -------------------------------------

        report_file = f"{model_name}_classification_report.txt"

        with open(report_file, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_file)

        # -------------------------------------
        # Log model
        # -------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(f"\n{model_name}")
        print(metrics)

        # -------------------------------------
        # Track best model
        # -------------------------------------

        if metrics["f1_score"] > best_f1:

            best_f1 = metrics["f1_score"]
            best_model = model
            best_model_name = model_name
            best_run_id = run.info.run_id

# =====================================================
# SAVE BEST MODEL LOCALLY
# =====================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print("\n==========================")
print(f"Best Model: {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")
print("==========================")

# =====================================================
# REGISTER BEST MODEL
# =====================================================

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="CreditRiskModel"
)

print(
    f"Registered model version: "
    f"{registered_model.version}"
)