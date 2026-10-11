from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

# Load dataset
X, y = load_iris(return_X_y=True)

# Train simple model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
model_path = Path("model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print(f"Model trained and saved as {model_path}")