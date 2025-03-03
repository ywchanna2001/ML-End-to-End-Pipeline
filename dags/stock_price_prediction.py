from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from airflow.models import Variable
import joblib
import os

# Constants
STOCK_SYMBOL = "AAPL"
LOOKBACK = 60  # Days of past data for ARIMA
MODEL_PATH = "/opt/airflow/saved_model/stock_model_arima.pkl"  # Adjust path if needed

# Default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 3, 3),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "stock_price_prediction",
    default_args=default_args,
    description="DAG for predicting Apple stock prices using ARIMA",
    schedule_interval="@daily",  # Change to '@weekly' if needed
    catchup=False,
)

# Step 1: Extract stock data
def get_stock_data(**kwargs):
    stock_data = yf.download(STOCK_SYMBOL, period="2y", interval="1d")
    close_prices = stock_data["Close"].tolist()  # Convert to list for XComs compatibility
    kwargs['ti'].xcom_push(key="stock_data", value=close_prices)

extract_task = PythonOperator(
    task_id="extract_stock_data",
    python_callable=get_stock_data,
    dag=dag,
)

# Step 2: Preprocess Data
def preprocess_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids="extract_stock_data", key="stock_data")
    
    # Use only the last 'LOOKBACK' days
    processed_data = data[-LOOKBACK:]
    ti.xcom_push(key="processed_data", value=processed_data)

preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

# Step 3: Train ARIMA Model
def train_arima_model(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids="preprocess_data", key="processed_data")
    
    # Convert list to Pandas Series for ARIMA
    data_series = pd.Series(data)

    # Train ARIMA Model
    model = ARIMA(data_series, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Save trained model using XComs
    ti.xcom_push(key="trained_model", value=model_fit)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_arima_model,
    dag=dag,
)

# Step 4: Evaluate Model
def evaluate_model(**kwargs):
    ti = kwargs['ti']
    model = ti.xcom_pull(task_ids="train_model", key="trained_model")
    data = ti.xcom_pull(task_ids="preprocess_data", key="processed_data")
    
    # Forecast next day's stock price
    forecast = model.forecast(steps=1)[0]
    real_price = data[-1]
    
    error = abs(real_price - forecast)
    
    print(f"Predicted next day's price: {forecast}")
    print(f"Real last day's price: {real_price}")
    print(f"Prediction error: {error}")

    # Save error as Airflow Variable (Optional)
    Variable.set("last_prediction_error", error)

evaluate_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)

# Step 5: Save the Model
def save_model(**kwargs):
    ti = kwargs['ti']
    model = ti.xcom_pull(task_ids="train_model", key="trained_model")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    print(f"Model saved at {MODEL_PATH}")

save_task = PythonOperator(
    task_id="save_model",
    python_callable=save_model,
    dag=dag,
)

# **DAG Task Flow**
extract_task >> preprocess_task >> train_task >> evaluate_task >> save_task
