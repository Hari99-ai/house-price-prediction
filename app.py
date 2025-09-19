import numpy as np
import pandas as pd
import os
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Initialize Flask app
app = Flask(__name__)

# Get the absolute path to the CSV file (relative path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "Housing.csv")

# Load the dataset
data = pd.read_csv(data_path)

# Prepare features and labels
X = data[["area", "bedrooms", "bathrooms", "stories"]]
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
linear_model = LinearRegression().fit(X_train, y_train)
decision_tree_model = DecisionTreeRegressor().fit(X_train, y_train)
gradient_boosting_model = GradientBoostingRegressor().fit(X_train, y_train)
xgboost_model = xgb.XGBRegressor().fit(X_train, y_train)
lightgbm_model = lgb.LGBMRegressor().fit(X_train, y_train)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from the form
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        model_type = request.form["model"]

        # Create feature array for prediction
        features = np.array([[area, bedrooms, bathrooms, stories]])

        # Predict based on selected model
        if model_type == "linear":
            prediction = linear_model.predict(features)[0]
        elif model_type == "tree":
            prediction = decision_tree_model.predict(features)[0]
        elif model_type == "gbm":
            prediction = gradient_boosting_model.predict(features)[0]
        elif model_type == "xgboost":
            prediction = xgboost_model.predict(features)[0]
        elif model_type == "lightgbm":
            prediction = lightgbm_model.predict(features)[0]
        else:
            raise ValueError("Invalid model type selected.")

        return render_template(
            "index.html", 
            prediction_text=f"Predicted House Price: ₹{round(prediction, 2)}"
        )
    except Exception as e:
        return render_template(
            "index.html", 
            prediction_text=f"Error occurred: {str(e)}"
        )

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
