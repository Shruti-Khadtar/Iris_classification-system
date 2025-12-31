# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_and_save(model_path="model_pipeline.pkl", meta_path="meta.pkl"):
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names.tolist()

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a pipeline: StandardScaler → LogisticRegression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=42
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train_model] Test accuracy: {acc:.4f}")

    # Save the model and metadata
    joblib.dump(pipeline, model_path)
    joblib.dump({"target_names": target_names}, meta_path)
    print(f"[train_model] Saved model to: {model_path}")
    print(f"[train_model] Saved metadata to: {meta_path}")

if __name__ == "__main__":
    train_and_save()
