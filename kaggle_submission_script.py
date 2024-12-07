# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from math import radians, sin, cos, sqrt, atan2


# At the beginning of the file, update the file discovery code:
input_dir = None
external_data_path = None
train_path = None
test_path = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        if filename == 'external_data.csv':
            external_data_path = filepath
        elif filename == 'train.parquet':
            train_path = filepath
        elif filename == 'final_test.parquet':  # Changed from test.parquet to final_test.parquet
            test_path = filepath
    if not input_dir and any(f.endswith('.parquet') for f in filenames):
        input_dir = dirname

# print(f"Found input directory: {input_dir}")
# print(f"Train data: {train_path}")
# print(f"Test data: {test_path}")
# print(f"External data: {external_data_path}")

def _encode_dates(X):
    """Encode datetime features."""
    X = X.copy()
    
    # Basic time features
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    
    # Additional time features
    X.loc[:, "is_weekend"] = (X["weekday"] >= 5).astype(int)
    X.loc[:, "is_rush_hour"] = ((X["hour"].isin([7,8,9]) | X["hour"].isin([17,18,19]))).astype(int)
    X.loc[:, "is_working_hour"] = ((X["hour"] >= 8) & (X["hour"] <= 18)).astype(int)
    
    # Cyclical encoding
    X.loc[:, "hour_sin"] = np.sin(2 * np.pi * X["hour"]/24)
    X.loc[:, "hour_cos"] = np.cos(2 * np.pi * X["hour"]/24)
    X.loc[:, "month_sin"] = np.sin(2 * np.pi * X["month"]/12)
    X.loc[:, "month_cos"] = np.cos(2 * np.pi * X["month"]/12)
    
    return X.drop(columns=["date"])

def _merge_external_data(X):
    """Merge input data with weather data."""
    X = X.copy()
    
    # Load and process weather data (using discovered path)
    weather_data = pd.read_csv(external_data_path, parse_dates=["date"])
    
    # Ensure both dataframes use nanosecond datetime
    X["date"] = pd.to_datetime(X["date"]).astype('datetime64[ns]')
    weather_data["date"] = pd.to_datetime(weather_data["date"]).astype('datetime64[ns]')
    
    # Process weather features
    weather_data.loc[:, "temp_celsius"] = weather_data["t"] - 273.15
    weather_data.loc[:, "is_cold"] = (weather_data["temp_celsius"] < 10).astype(int)
    weather_data.loc[:, "is_hot"] = (weather_data["temp_celsius"] > 25).astype(int)
    
    wind_kmh = weather_data["ff"] * 3.6
    weather_data.loc[:, "high_wind"] = (wind_kmh > 20).astype(int)
    
    weather_data.loc[:, "is_raining"] = (weather_data["rr1"] > 0).astype(int)
    weather_data.loc[:, "heavy_rain"] = (weather_data["rr1"] > 5).astype(int)
    
    weather_data.loc[:, "poor_visibility"] = (weather_data["vv"] < 10000).astype(int)
    weather_data.loc[:, "high_humidity"] = (weather_data["u"] > 80).astype(int)
    
    # Keep only necessary columns
    cols_to_keep = ["date", "temp_celsius", "is_cold", "is_hot", "high_wind",
                    "is_raining", "heavy_rain", "poor_visibility", "high_humidity"]
    weather_df = weather_data[cols_to_keep]
    
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    
    # Merge with weather data
    X = pd.merge_asof(
        X.sort_values("date"),
        weather_df.sort_values("date"),
        on="date",
        direction="nearest"
    )
    
    # Sort back to original order and clean up
    X = X.sort_values("orig_index")
    del X["orig_index"]
    
    return X

def _add_location_features(X):
    """Add location-based features."""
    X = X.copy()
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    PARIS_CENTER_LAT = 48.8566
    PARIS_CENTER_LON = 2.3522
    X.loc[:, "distance_to_center"] = X.apply(
        lambda row: haversine_distance(
            row["latitude"], row["longitude"], 
            PARIS_CENTER_LAT, PARIS_CENTER_LON
        ), axis=1
    )
    
    return X

def prepare_features(data):
    """Prepare features for model input."""
    X = data.drop(["counter_id", "site_id", "counter_technical_id", 
                   "coordinates", "counter_installation_date"], axis=1)
    if "log_bike_count" in X.columns:
        X = X.drop(["log_bike_count", "bike_count"], axis=1)
    return X

def get_estimator():
    """Create and return the full prediction pipeline."""
    # Get date columns for encoding
    date_cols = ["year", "month", "day", "weekday", "hour", 
                 "is_weekend", "is_rush_hour", "is_working_hour",
                 "hour_sin", "hour_cos", "month_sin", "month_cos"]
    
    # Define categorical columns
    categorical_cols = ["counter_name", "site_name"]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    
    # Create the full pipeline
    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        FunctionTransformer(_add_location_features, validate=False),
        FunctionTransformer(_encode_dates, validate=False),
        preprocessor,
        XGBRegressor(random_state=42)
    )
    
    return pipe

def main():
    """Main function to generate Kaggle submission."""
    # Load data using discovered paths
    train_data = pd.read_parquet(train_path)
    test_data = pd.read_parquet(test_path)
    
    # Prepare features
    X_train = prepare_features(train_data)
    y_train = train_data["log_bike_count"].values
    X_test = prepare_features(test_data)
    
    # Create and train model
    pipe = get_estimator()
    pipe.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = pipe.predict(X_test)
    
    # Create submission file
    results = pd.DataFrame({
        "Id": np.arange(y_pred.shape[0]),
        "log_bike_count": y_pred
    })
    
    # Save submission
    results.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    main()