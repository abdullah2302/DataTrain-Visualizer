from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
CORS(app)

@app.route('/train', methods=['POST', 'GET'])
def train_model():
    if request.method == 'GET':
        return jsonify({'message': 'Use POST to train a model'}), 200

    try:
        if 'file' not in request.files or 'target' not in request.form or 'model' not in request.form or 'graph' not in request.form:
            return jsonify({'error': 'Missing required fields (file, target, model, or graph)'}), 400

        file = request.files['file']
        target_col = request.form['target']
        model_type = request.form['model']
        graph_type = request.form['graph']
        train_ratio = float(request.form.get('train_ratio', 0.8))  # Default 80% train
        max_columns = int(request.form.get('max_columns', 500))  # Default 500 columns

        # Load CSV with proper encoding handling
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except Exception as e:
            logging.error(f"Error reading CSV: {str(e)}")
            return jsonify({'error': 'Invalid CSV format or encoding issue'}), 400

        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        if target_col not in df.columns:
            return jsonify({'error': f'Target column "{target_col}" not found in CSV'}), 400

        # Convert categorical target variable
        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Limit columns if exceeding max_columns
        if X.shape[1] > max_columns:
            X = X.iloc[:, :max_columns]  # Take first max_columns columns

        # Convert categorical features to numeric using one-hot encoding
        X = pd.get_dummies(X, drop_first=True)

        # Handle missing values (NaNs) by filling with column mean
        X = X.fillna(X.mean())
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            y = pd.to_numeric(y, errors='coerce').fillna(0)  # Convert y to numeric, fill NaN with 0

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

        # Standardize data for SVM and K-Means
        scaler = StandardScaler()
        if model_type in ['svm_regressor', 'svm_classifier', 'kmeans']:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            # Ensure numeric data for other models
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)

        # Model selection and configuration
        models = {
            'linear_regression': LinearRegression(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest_regressor': RandomForestRegressor(random_state=42),
            'random_forest_classifier': RandomForestClassifier(random_state=42),
            'svm_regressor': SVR(kernel='rbf'),
            'svm_classifier': SVC(kernel='rbf', probability=True),
            'kmeans': KMeans(n_clusters=3, random_state=42, n_init=10)
        }

        if model_type not in models:
            return jsonify({'error': 'Invalid model type'}), 400

        model = models[model_type]

        # Train model and get predictions
        if model_type != 'kmeans':
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
        else:
            model.fit(X_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

        # Ensure predictions are numeric
        train_predictions = np.nan_to_num(train_predictions, nan=0)
        test_predictions = np.nan_to_num(test_predictions, nan=0)

        # Extract X-axis data (use first column if available, else use indices)
        def get_x_data(df_or_array):
            if isinstance(df_or_array, pd.DataFrame):
                if df_or_array.shape[1] > 0:
                    return df_or_array.iloc[:, 0].tolist()  # Use first column if DataFrame
                return [i for i in range(len(df_or_array))]  # Use indices if no columns
            elif isinstance(df_or_array, np.ndarray):
                if df_or_array.shape[1] > 0:
                    return df_or_array[:, 0].tolist()  # Use first column if NumPy array
                return [i for i in range(len(df_or_array))]  # Use indices if no columns
            return [i for i in range(len(df_or_array))]  # Fallback to indices

        # Prepare response with train and test data
        result = {
            'train_predictions': train_predictions.tolist(),
            'train_x_data': get_x_data(X_train),
            'train_y_data': y_train.tolist(),
            'test_predictions': test_predictions.tolist(),
            'test_x_data': get_x_data(X_test),
            'test_y_data': y_test.tolist(),
            'graph_type': graph_type
        }
        return jsonify(result)

    except Exception as e:
        logging.error(f"Server Error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=False)