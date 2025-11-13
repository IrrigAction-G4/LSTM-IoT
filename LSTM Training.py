import pandas as pd
import numpy as np
import tensorflow as tf
import meteostat
import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from meteostat import Point, Daily
from datetime import datetime
import pickle
import keras_tuner as kt

# Define the directory where you want the models to be saved (follow the format)
save_dir = r"C:\Users\DeviceName\path1\path2\path3\onwards"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving files to: {save_dir}")

location = Point(14.8527, 120.8160) # Malolos, Philippines (replace with desired location - focus of your study)

start = datetime(2019, 1, 1) # Start date for historical data - can be adjusted
end = datetime.today()

print("Fetching historical weather data...")
data = Daily(location, start, end)
data = data.fetch()

csv_file_name = os.path.join(save_dir, "meteostat_malolos_weather_data.csv") # You can change the file name if needed (ensure na .csv ang ending)
data.to_csv(csv_file_name)
print(f"Weather data has been saved to {csv_file_name}")

weather = pd.read_csv(csv_file_name, index_col="time")
weather.index = pd.to_datetime(weather.index)

weather = weather.ffill()

def calculate_dewpoint(temp, humidity):
    if pd.isna(temp) or pd.isna(humidity):
        return np.nan
    a = 17.27
    b = 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
    return (b * alpha) / (a - alpha)

def calculate_heat_index(temp, humidity):
    if pd.isna(temp) or pd.isna(humidity):
        return np.nan
    temp_f = (temp * 9 / 5) + 32
    hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
    if hi > 80:
        hi = -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
        hi = hi - 0.22475541 * temp_f * humidity
        hi = hi - 6.83783e-3 * temp_f ** 2
        hi = hi - 5.481717e-2 * humidity ** 2
        hi = hi + 1.22874e-3 * temp_f ** 2 * humidity
        hi = hi + 8.5282e-4 * temp_f * humidity ** 2
        hi = hi - 1.99e-6 * temp_f ** 2 * humidity ** 2
    return (hi - 32) * 5 / 9

if 'rhum' in weather.columns and 'tavg' in weather.columns:
    weather['dewpoint'] = weather.apply(lambda row: calculate_dewpoint(row['tavg'], row['rhum']), axis=1)
    weather['heat_index'] = weather.apply(lambda row: calculate_heat_index(row['tavg'], row['rhum']), axis=1)

weather['temp_range'] = weather['tmax'] - weather['tmin']
weather['pres_delta_1d'] = weather['pres'].diff(1)
weather['pres_delta_3d'] = weather['pres'].diff(3)
weather['pres_rolling_trend'] = weather['pres'].rolling(window=3).apply(
    lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
)

weather['is_wet_season'] = weather.index.month.isin([6, 7, 8, 9, 10, 11]).astype(int)

weather['consecutive_dry_days'] = 0
dry_day_count = 0

for i, row in weather.iterrows():
    if row['prcp'] < 0.5:
        dry_day_count += 1
    else:
        dry_day_count = 0
    weather.at[i, 'consecutive_dry_days'] = dry_day_count

weather['day_of_year'] = weather.index.dayofyear
weather['month'] = weather.index.month
weather['day_of_month'] = weather.index.day

weather['sin_day'] = np.sin(2 * np.pi * weather['day_of_year'] / 365.25)
weather['cos_day'] = np.cos(2 * np.pi * weather['day_of_year'] / 365.25)
weather['sin_month'] = np.sin(2 * np.pi * weather['month'] / 12)
weather['cos_month'] = np.cos(2 * np.pi * weather['month'] / 12)

def classify_rain(prcp, pres, pres_delta, consecutive_dry, is_wet_season, tavg, tmin, wspd, wdir):
    if prcp > 0:
        if prcp >= 15 or (pres < 1005 and wspd > 20):
            return 3
        elif prcp >= 5 or (pres < 1010 and wspd > 15) or (pres_delta < -2 and is_wet_season):
            return 2
        else:
            return 1
    if pres_delta < -3 and is_wet_season and consecutive_dry < 5 and wspd > 10:
        return 1
    return 0

weather["rain_intensity"] = weather.apply(
    lambda row: classify_rain(
        row["prcp"],
        row["pres"],
        row.get("pres_delta_1d", 0),
        row["consecutive_dry_days"],
        row["is_wet_season"],
        row["tavg"],
        row["tmin"],
        row["wspd"],
        row["wdir"]
    ), axis=1)
weather["rain"] = (weather["prcp"] > 0).astype(int)

for col in ['prcp', 'tavg', 'tmin', 'tmax', 'wspd', 'pres', 'dewpoint', 'heat_index', 'temp_range']:
    if col in weather.columns:
        for lag in [1, 3, 7]:
            weather[f'{col}_lag_{lag}'] = weather[col].shift(lag)

for col in ['prcp', 'tavg', 'tmin', 'tmax', 'wspd', 'pres']:
    weather[f'{col}_rolling_mean_7'] = weather[col].rolling(window=7).mean()
    weather[f'{col}_rolling_std_7'] = weather[col].rolling(window=7).std()

weather = weather.fillna(method='bfill').fillna(method='ffill')

print(f"Shape after adding features and handling NaNs: {weather.shape}")

processed_csv_path = os.path.join(save_dir, "processed_weather_data.csv")
weather.to_csv(processed_csv_path)
print(f"Processed data saved to '{processed_csv_path}'")

scaler = MinMaxScaler()
weather_scaled = scaler.fit_transform(weather)

scaler_path = os.path.join(save_dir, "weather_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to '{scaler_path}'")

sequence_length = 15

def create_sequences(data, target_column):
    x_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        x_seq.append(data[i:i + sequence_length])
        y_seq.append(data[i + sequence_length, target_column])
    return np.array(x_seq), np.array(y_seq)

columns = list(weather.columns)
prcp_col = columns.index("prcp")
tmin_col = columns.index("tmin")
tmax_col = columns.index("tmax")
rain_col = columns.index("rain")
rain_intensity_col = columns.index("rain_intensity")

column_mapping = {
    'prcp_col': prcp_col,
    'tmin_col': tmin_col,
    'tmax_col': tmax_col,
    'rain_col': rain_col,
    'rain_intensity_col': rain_intensity_col,
    'columns': columns
}

mapping_path = os.path.join(save_dir, "column_mapping.pkl")
with open(mapping_path, 'wb') as f:
    pickle.dump(column_mapping, f)
print(f"Column mapping saved to '{mapping_path}'")

x_data, y_prcp = create_sequences(weather_scaled, prcp_col)
_, y_tmin = create_sequences(weather_scaled, tmin_col)
_, y_tmax = create_sequences(weather_scaled, tmax_col)
_, y_rain = create_sequences(weather_scaled, rain_col)
_, y_rain_intensity = create_sequences(weather_scaled, rain_intensity_col)

split = int(0.8 * len(x_data))
x_train, x_test = x_data[:split], x_data[split:]
y_prcp_train, y_prcp_test = y_prcp[:split], y_prcp[split:]
y_tmin_train, y_tmin_test = y_tmin[:split], y_tmin[split:]
y_tmax_train, y_tmax_test = y_tmax[:split], y_tmax[split:]
y_rain_train, y_rain_test = y_rain[:split], y_rain[split:]
y_rain_intensity_train, y_rain_intensity_test = y_rain_intensity[:split], y_rain_intensity[split:]

# New hypermodel builder function for keras-tuner - simplified for faster tuning
def build_tunable_lstm_model(hp, output_units=1, activation="linear", input_shape=None):
    if input_shape is None:
        input_shape = (sequence_length, x_data.shape[2])

    model = Sequential()
    
    lstm_units_1 = hp.Choice('lstm_units_1', values=[32, 64, 128])
    model.add(LSTM(lstm_units_1, return_sequences=True, input_shape=input_shape))
    
    model.add(Dropout(0.3))
    
    lstm_units_2 = hp.Choice('lstm_units_2', values=[32, 64])
    model.add(LSTM(lstm_units_2, return_sequences=False))
    
    dense_units = hp.Choice('dense_units', values=[16, 32])
    model.add(Dense(dense_units, activation='relu'))
    
    model.add(Dense(output_units, activation=activation))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3])

    if output_units > 1:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    elif activation == 'sigmoid':
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'mae'
        metrics = []

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

# Create model builders for different prediction targets
def create_prcp_hypermodel(hp):
    return build_tunable_lstm_model(hp)

def create_tmin_hypermodel(hp):
    return build_tunable_lstm_model(hp)

def create_tmax_hypermodel(hp):
    return build_tunable_lstm_model(hp)

def create_rain_hypermodel(hp):
    return build_tunable_lstm_model(hp, output_units=1, activation='sigmoid')

def create_rain_intensity_hypermodel(hp):
    return build_tunable_lstm_model(hp, output_units=4, activation='softmax')

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,  
    restore_best_weights=True
)

batch_size_choice = [32, 64]

print("Performing streamlined hyperparameter tuning...")

tuner_results = {}
best_hyperparameters = {}
best_models = {}

# Function to perform fast hyperparameter tuning for each model - can be further adjusted
def tune_model_fast(create_hypermodel_fn, x_train, y_train, model_name):
    print(f"\nQuick tuning for {model_name} model...")
    
    tuner = kt.RandomSearch(
        create_hypermodel_fn,
        objective='val_loss',
        max_trials=5,  
        executions_per_trial=1,
        directory=os.path.join(save_dir, 'tuning_results'),
        project_name=f'{model_name}_quick_tuning'
    )
    
    # Use a small subset of the data for faster tuning
    train_size = min(len(x_train), 500)  
    x_tune = x_train[:train_size]
    y_tune = y_train[:train_size]
    
    # Quick tuning - can be modified as needed
    tuner.search(
        x_tune, y_tune,
        epochs=10,  
        validation_split=0.2,
        callbacks=[early_stopping],
        batch_size=32,  
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(1)[0]
    
    model = tuner.hypermodel.build(best_hps)
    
    # Train the model on the full dataset - can be modified as needed
    history = model.fit(
        x_train, y_train,
        epochs=15, 
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save results
    tuner_results[model_name] = {
        'best_val_loss': min(history.history['val_loss'])
    }
    best_hyperparameters[model_name] = best_hps.values
    best_models[model_name] = model
    
    print(f"Best hyperparameters for {model_name}:")
    for param, value in best_hps.values.items():
        print(f"  {param}: {value}")
    
    return model

print("Starting quick hyperparameter tuning process...")
print("This should complete in minutes instead of hours.")

rain_model = tune_model_fast(create_rain_hypermodel, x_train, y_rain_train, "rain")
rain_intensity_model = tune_model_fast(create_rain_intensity_hypermodel, x_train, y_rain_intensity_train, "rain_intensity")
prcp_model = tune_model_fast(create_prcp_hypermodel, x_train, y_prcp_train, "prcp")
tmin_model = tune_model_fast(create_tmin_hypermodel, x_train, y_tmin_train, "tmin")
tmax_model = tune_model_fast(create_tmax_hypermodel, x_train, y_tmax_train, "tmax")

# Save best hyperparameters
hyperparams_path = os.path.join(save_dir, "best_hyperparameters.pkl")
with open(hyperparams_path, 'wb') as f:
    pickle.dump(best_hyperparameters, f)
print(f"Best hyperparameters saved to '{hyperparams_path}'")

# Rest of the evaluation code
def evaluate_model(model, x_test, y_test, target_column, model_name):
    y_pred_norm = model.predict(x_test, verbose=0)

    pred_matrix = np.zeros((len(y_pred_norm), len(columns)))
    pred_matrix[:, target_column] = y_pred_norm.flatten()

    actual_matrix = np.zeros((len(y_test), len(columns)))
    actual_matrix[:, target_column] = y_test

    y_pred = scaler.inverse_transform(pred_matrix)[:, target_column]
    y_true = scaler.inverse_transform(actual_matrix)[:, target_column]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{model_name} Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return mae, rmse, y_pred, y_true

print("\nEvaluating model performance on test data...")
tmin_mae, tmin_rmse, tmin_pred, tmin_true = evaluate_model(tmin_model, x_test, y_tmin_test, tmin_col,
                                                          "Minimum Temperature")
tmax_mae, tmax_rmse, tmax_pred, tmax_true = evaluate_model(tmax_model, x_test, y_tmax_test, tmax_col,
                                                          "Maximum Temperature")

rain_pred_prob = rain_model.predict(x_test, verbose=0).flatten()
rain_pred = (rain_pred_prob > 0.5).astype(int)
rain_accuracy = np.mean(rain_pred == y_rain_test)
print("\nRain Occurrence Evaluation:")
print(f"Accuracy: {rain_accuracy:.4f}")

rain_intensity_pred_prob = rain_intensity_model.predict(x_test, verbose=0)
rain_intensity_pred = np.argmax(rain_intensity_pred_prob, axis=1)
rain_intensity_accuracy = np.mean(rain_intensity_pred == y_rain_intensity_test)
print("\nRain Intensity Evaluation:")
print(f"Accuracy: {rain_intensity_accuracy:.4f}")

prcp_mae, prcp_rmse, prcp_pred, prcp_true = evaluate_model(prcp_model, x_test, y_prcp_test, prcp_col,
                                                          "Precipitation (Tuned LSTM)")

# Save models
prcp_model_path = os.path.join(save_dir, 'prcp_model.keras')
tmin_model_path = os.path.join(save_dir, 'tmin_model.keras')
tmax_model_path = os.path.join(save_dir, 'tmax_model.keras')
rain_model_path = os.path.join(save_dir, 'rain_model.keras')
rain_intensity_model_path = os.path.join(save_dir, 'rain_intensity_model.keras')

try:
    prcp_model.save(prcp_model_path)
    print(f"Precipitation model saved to {prcp_model_path}")

    tmin_model.save(tmin_model_path)
    print(f"Minimum temperature model saved to {tmin_model_path}")

    tmax_model.save(tmax_model_path)
    print(f"Maximum temperature model saved to {tmax_model_path}")

    rain_model.save(rain_model_path)
    print(f"Rain occurrence model saved to {rain_model_path}")

    rain_intensity_model.save(rain_intensity_model_path)
    print(f"Rain intensity model saved to {rain_intensity_model_path}")
except Exception as e:
    print(f"Error saving models: {str(e)}")
    try:
        prcp_model.save(os.path.join(save_dir, 'prcp_model.h5'))
        tmin_model.save(os.path.join(save_dir, 'tmin_model.h5'))
        tmax_model.save(os.path.join(save_dir, 'tmax_model.h5'))
        rain_model.save(os.path.join(save_dir, 'rain_model.h5'))
        rain_intensity_model.save(os.path.join(save_dir, 'rain_intensity_model.h5'))
        print("All models saved successfully in h5 format")
    except Exception as e2:
        print(f"Error saving models in h5 format: {str(e2)}")

model_config = {
    "use_two_stage": False,
    "use_convlstm": False,
    "sequence_length": sequence_length,
    "model_files": {
        "rain": "rain_model.keras",
        "prcp": "prcp_model.keras",
        "tmin": "tmin_model.keras",
        "tmax": "tmax_model.keras",
        "rain_intensity": "rain_intensity_model.keras"
    }
}

model_config_path = os.path.join(save_dir, "model_config.pkl")
with open(model_config_path, 'wb') as f:
    pickle.dump(model_config, f)
print(f"Model configuration saved to '{model_config_path}'")

# Save tuner results
tuner_results_path = os.path.join(save_dir, "tuner_results.pkl")
with open(tuner_results_path, 'wb') as f:
    pickle.dump(tuner_results, f)
print(f"Tuner results saved to '{tuner_results_path}'")

evaluation = {
    "Precipitation (Tuned LSTM)": {"MAE": prcp_mae, "RMSE": prcp_rmse},
    "Minimum Temperature": {"MAE": tmin_mae, "RMSE": tmin_rmse},
    "Maximum Temperature": {"MAE": tmax_mae, "RMSE": tmax_rmse},
    "Rain Occurrence": {"Accuracy": rain_accuracy},
    "Rain Intensity": {"Accuracy": rain_intensity_accuracy}
}

evaluation_data = {
    'prcp_true': prcp_true,
    'prcp_pred': prcp_pred,
    'tmin_true': tmin_true,
    'tmin_pred': tmin_pred,
    'tmax_true': tmax_true,
    'tmax_pred': tmax_pred,
    'test_indices': list(range(split, len(weather) - sequence_length))
}

eval_data_path = os.path.join(save_dir, 'evaluation_data.pkl')
with open(eval_data_path, 'wb') as f:
    pickle.dump(evaluation_data, f)
print(f"Evaluation data saved to '{eval_data_path}'")

metrics_path = os.path.join(save_dir, "model_evaluation_metrics.csv")
pd.DataFrame(evaluation).to_csv(metrics_path)
print(f"Model evaluation metrics saved to '{metrics_path}'")

print("Hyperparameter tuning, training and evaluation completed successfully!")