import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Load the dataset
data = pd.read_csv("C:/Users/hari9/Downloads/Housing.csv")

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

# Initialize Flask app
app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
