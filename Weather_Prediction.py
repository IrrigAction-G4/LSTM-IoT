import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import json

# Insert Directory Path where you saved the models - refer kayo dun sa LSTM Training.py (follow the format)
model_dir = 'C:\\Users\\DeviceName\\path1\\path2\\path3\\onwards'
# Insert Directory Path for predictions
predictions_dir = 'C:\\Users\\DeviceName\\path1\\path2\\path3\\onwards'
# Create the predictions directory if it doesn't exist
os.makedirs(predictions_dir, exist_ok=True)

def load_models_and_data():
    """Load all required models and data"""
    try:
        # Load models
        prcp_model = load_model(os.path.join(model_dir, 'prcp_model.keras'))
        tmin_model = load_model(os.path.join(model_dir, 'tmin_model.keras'))
        tmax_model = load_model(os.path.join(model_dir, 'tmax_model.keras'))
        rain_model = load_model(os.path.join(model_dir, 'rain_model.keras'))
        rain_intensity_model = load_model(os.path.join(model_dir, 'rain_intensity_model.keras'))

        with open(os.path.join(model_dir, 'weather_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        with open(os.path.join(model_dir, 'column_mapping.pkl'), 'rb') as f:
            column_mapping = pickle.load(f)

        # Load historical weather data
        weather = pd.read_csv(os.path.join(model_dir, 'processed_weather_data.csv'), index_col='time')
        weather.index = pd.to_datetime(weather.index)
        weather_scaled = scaler.transform(weather)

        return {
            'models': {
                'prcp': prcp_model,
                'tmin': tmin_model,
                'tmax': tmax_model,
                'rain': rain_model,
                'rain_intensity': rain_intensity_model
            },
            'scaler': scaler,
            'column_mapping': column_mapping,
            'weather_scaled': weather_scaled,
            'weather': weather
        }
    except Exception as e:
        return {'error': f'Error loading models and data: {str(e)}'}

def get_enhanced_rain_predictions(precipitation, precip_type=None, precip_intensity=None, precip_probability=None):
    """Calculate enhanced rain predictions based on precipitation data"""
    if precip_type is not None and precip_probability is not None:
        probability = precip_probability

        if precip_type.lower() == "none" or probability < 0.20:
            rain_category = "No (Very Low Chance)"
        elif probability < 0.50:
            rain_category = "Low Chance"
        else:
            rain_category = "Yes (High Chance)"

        if precip_intensity is not None and rain_category == "High Chance (Yes)":
            intensity = precip_intensity.capitalize()
        elif rain_category == "High Chance (Yes)":
            if precipitation < 2.5:
                intensity = "Light"
            elif precipitation < 10:
                intensity = "Moderate"
            else:
                intensity = "Heavy"
        else:
            intensity = "None"
    else:
        probability = min(1.0, precipitation / 10) if precipitation > 0 else 0

        if probability < 0.20:
            rain_category = "No (Very Low Chance)"
            intensity = "None"
        elif probability < 0.50:
            rain_category = "Low Chance"
            if precipitation < 2.5:
                intensity = "Light"
            else:
                intensity = "Moderate"
        else:
            rain_category = "High Chance (Yes)"
            if precipitation < 2.5:
                intensity = "Light"
            elif precipitation < 10:
                intensity = "Moderate"
            else:
                intensity = "Heavy"

    return rain_category, intensity, probability

# Fetch AccuWeather Data - more accurate if included (remove if nahihirapan kayo iintegrate)
def fetch_accuweather_data(api_key, location_key):
    """Fetch weather data from AccuWeather API"""
    base_url = "http://dataservice.accuweather.com"
    forecast_url = f"{base_url}/forecasts/v1/daily/5day/{location_key}?apikey={api_key}&metric=true&details=true"

    try:
        print(f"Fetching AccuWeather data from: {forecast_url}")
        response = requests.get(forecast_url)
        response.raise_for_status()
        data = response.json()
        
        print(f"AccuWeather API Response Headers: {response.headers}")
        print(f"AccuWeather API Response Status: {response.status_code}")

        if "DailyForecasts" not in data:
            print("No DailyForecasts in AccuWeather response")
            return None
            
        print(f"Number of days in forecast: {len(data['DailyForecasts'])}")
        
        forecast_data = []
        for day in data["DailyForecasts"]:
            date_str = day["Date"]
            date_obj = pd.Timestamp(date_str) 
            formatted_date = date_obj.strftime("%Y-%m-%d")

            min_temp = day["Temperature"]["Minimum"]["Value"]
            max_temp = day["Temperature"]["Maximum"]["Value"]

            day_icon_phrase = day["Day"].get("IconPhrase", "Unknown")
            night_icon_phrase = day["Night"].get("IconPhrase", "Unknown")
            
            print(f"Date: {formatted_date}")
            print(f"Day IconPhrase: {day_icon_phrase}")
            print(f"Night IconPhrase: {night_icon_phrase}")

            day_has_precip = day["Day"].get("HasPrecipitation", False)
            day_precip_type = day["Day"].get("PrecipitationType", "None")
            day_precip_intensity = day["Day"].get("PrecipitationIntensity", "None")
            day_precip_probability = day["Day"].get("PrecipitationProbability", 0) / 100.0

            night_has_precip = day["Night"].get("HasPrecipitation", False)
            night_precip_type = day["Night"].get("PrecipitationType", "None")
            night_precip_intensity = day["Night"].get("PrecipitationIntensity", "None")
            night_precip_probability = day["Night"].get("PrecipitationProbability", 0) / 100.0

            day_estimated_precip = calculate_precipitation_estimate(day_precip_type, day_has_precip, 
                                                                 day_precip_intensity, day_precip_probability)
            night_estimated_precip = calculate_precipitation_estimate(night_precip_type, night_has_precip, 
                                                                    night_precip_intensity, night_precip_probability)
            total_precip = day_estimated_precip + night_estimated_precip

            forecast_data.append({
                'time': date_obj,  
                'prcp': total_precip,
                'day_prcp': day_estimated_precip,
                'night_prcp': night_estimated_precip,
                'tmin': min_temp,
                'tmax': max_temp,
                'day_precip_type': day_precip_type,
                'day_precip_intensity': day_precip_intensity,
                'day_precip_probability': day_precip_probability,
                'night_precip_type': night_precip_type,
                'night_precip_intensity': night_precip_intensity,
                'night_precip_probability': night_precip_probability,
                'day_weather': day_icon_phrase,  
                'night_weather': night_icon_phrase  
            })

        return pd.DataFrame(forecast_data).set_index('time')

    except Exception as e:
        print(f"Error fetching AccuWeather data: {str(e)}")
        return None

def calculate_precipitation_estimate(precip_type, has_precip, precip_intensity, precip_probability):
    """Calculate precipitation estimate based on type and intensity"""
    if precip_type.lower() == "none" or not has_precip:
        return 0.0
    elif precip_intensity.lower() == "light":
        return 2.0 * precip_probability
    elif precip_intensity.lower() == "moderate":
        return 5.0 * precip_probability
    elif precip_intensity.lower() == "heavy":
        return 15.0 * precip_probability
    else:
        return 0.0

def make_lstm_predictions(weather_scaled, models, scaler, column_mapping, api_dates=None):
    """Make LSTM predictions for the next 5 days"""
    sequence_length = 15
    future_input = weather_scaled[-sequence_length:].copy()
    input_sequence = future_input.copy()
    predictions = []

    prcp_col = column_mapping['prcp_col']
    tmin_col = column_mapping['tmin_col']
    tmax_col = column_mapping['tmax_col']
    rain_col = column_mapping['rain_col']
    rain_intensity_col = column_mapping['rain_intensity_col']
    columns = column_mapping['columns']

    # Determine prediction dates (range of dates can be changed)
    if api_dates is None:
        current_date = pd.Timestamp.today().normalize()
        pred_dates = [current_date + pd.Timedelta(days=i+1) for i in range(5)]
    else:
        pred_dates = api_dates[:5]

    for i, pred_date in enumerate(pred_dates):
        current_sequence = input_sequence.reshape(1, sequence_length, -1)

        # Make predictions
        prcp_pred = float(models['prcp'].predict(current_sequence, verbose=0)[0][0])
        tmin_pred = float(models['tmin'].predict(current_sequence, verbose=0)[0][0])
        tmax_pred = float(models['tmax'].predict(current_sequence, verbose=0)[0][0])
        rain_prob = float(models['rain'].predict(current_sequence, verbose=0)[0][0])
        rain_intensity_probs = models['rain_intensity'].predict(current_sequence, verbose=0)[0]
        rain_intensity_pred = int(np.argmax(rain_intensity_probs))

        next_day = prepare_next_day_features(input_sequence, prcp_pred, tmin_pred, tmax_pred, 
                                           rain_prob, rain_intensity_pred, pred_date, columns, 
                                           prcp_col, tmin_col, tmax_col, rain_col, rain_intensity_col, i)

        pred_matrix = np.zeros((1, len(columns)))
        pred_matrix[0, prcp_col] = prcp_pred
        pred_matrix[0, tmin_col] = tmin_pred
        pred_matrix[0, tmax_col] = tmax_pred

        actual_values = scaler.inverse_transform(pred_matrix)[0]

        predictions.append(create_prediction_dict(pred_date, actual_values, prcp_col, tmin_col, tmax_col))

        input_sequence = np.vstack([input_sequence[1:], next_day])

    return predictions

def prepare_next_day_features(input_sequence, prcp_pred, tmin_pred, tmax_pred, 
                            rain_prob, rain_intensity_pred, pred_date, columns,
                            prcp_col, tmin_col, tmax_col, rain_col, rain_intensity_col, i):
    """Prepare features for the next day's prediction"""
    next_day = np.zeros_like(input_sequence[0])
    next_day[:] = input_sequence[-1]

    next_day[prcp_col] = prcp_pred
    next_day[tmin_col] = tmin_pred
    next_day[tmax_col] = tmax_pred
    next_day[rain_col] = 1 if rain_prob > 0.5 else 0
    next_day[rain_intensity_col] = rain_intensity_pred

    day_of_year = pred_date.dayofyear
    month = pred_date.month
    day_of_month = pred_date.day

    col_indices = {col: columns.index(col) for col in ['day_of_year', 'month', 'day_of_month',
                                                       'sin_day', 'cos_day', 'sin_month', 'cos_month']}

    next_day[col_indices['day_of_year']] = day_of_year / 365.25
    next_day[col_indices['month']] = month / 12
    next_day[col_indices['day_of_month']] = day_of_month / 31
    next_day[col_indices['sin_day']] = np.sin(2 * np.pi * day_of_year / 365.25)
    next_day[col_indices['cos_day']] = np.cos(2 * np.pi * day_of_year / 365.25)
    next_day[col_indices['sin_month']] = np.sin(2 * np.pi * month / 12)
    next_day[col_indices['cos_month']] = np.cos(2 * np.pi * month / 12)

    set_lag_features(next_day, input_sequence, columns, i)

    return next_day

def set_lag_features(next_day, input_sequence, columns, i):
    """Set lag features for the next day's prediction"""
    if i > 0:
        for col in ['prcp', 'tmin', 'tmax', 'tavg', 'wspd', 'pres']:
            lag_col = f'{col}_lag_1'
            if lag_col in columns:
                lag_idx = columns.index(lag_col)
                orig_idx = columns.index(col)
                next_day[lag_idx] = input_sequence[-1][orig_idx]

        if i >= 3:
            for col in ['prcp', 'tmin', 'tmax', 'tavg', 'wspd', 'pres']:
                lag_col = f'{col}_lag_3'
                if lag_col in columns:
                    lag_idx = columns.index(lag_col)
                    orig_idx = columns.index(col)
                    next_day[lag_idx] = input_sequence[-3][orig_idx]

def create_prediction_dict(pred_date, actual_values, prcp_col, tmin_col, tmax_col):
    """Create a dictionary for a single prediction"""
    real_prcp = round(max(0, actual_values[prcp_col]), 2)
    real_tmin = round(actual_values[tmin_col], 2)
    real_tmax = round(actual_values[tmax_col], 2)

    rain_category, rain_intensity, rain_prob_value = get_enhanced_rain_predictions(real_prcp)
    rain_prob_value = round(rain_prob_value, 2)

    return {
        "Date": pred_date,
        "Precipitation (mm)": real_prcp,
        "Tmin": real_tmin,
        "Tmax": real_tmax,
        "Rain Probability": rain_prob_value,
        "Rain Category": rain_category,
        "Rain Intensity": rain_intensity,
        "Precipitation Type": "Rain" if "High Chance" in rain_category else "None"
    }

def blend_predictions(api_data, model_predictions, blend_weights):
    """Blend API and model predictions"""
    blended_predictions = []

    api_dates = set(api_data.index)
    
    for i, pred in enumerate(model_predictions):
        pred_date = pred['Date'] if isinstance(pred['Date'], pd.Timestamp) else pd.Timestamp(pred['Date'])
        
        if pred_date in api_dates:
            blended_pred = blend_single_prediction(pred, api_data, pred_date, blend_weights, i)
            blended_predictions.append(blended_pred)
        else:
            blended_predictions.append(create_fallback_prediction(pred))

    return blended_predictions

def blend_single_prediction(pred, api_data, pred_date, blend_weights, i):
    """Blend a single prediction with API data"""
    blended_pred = pred.copy()
    day_weights = blend_weights.get(i, blend_weights.get('default'))

    day_prob = round(api_data.loc[pred_date, 'day_precip_probability'], 2)
    night_prob = round(api_data.loc[pred_date, 'night_precip_probability'], 2)
    overall_prob = max(day_prob, night_prob)

    api_prcp = round(api_data.loc[pred_date, 'prcp'], 2)
    model_prcp = round(pred['Precipitation (mm)'], 2)
    weight = day_weights['prcp']

    blended_pred['Precipitation (mm)'] = round(blend_precipitation(overall_prob, api_prcp, model_prcp, weight), 2)
    blended_pred['Day Precipitation (mm)'] = round(blend_day_precipitation(day_prob, api_data, pred_date, model_prcp, weight), 2)
    blended_pred['Night Precipitation (mm)'] = round(blend_night_precipitation(night_prob, api_data, pred_date, model_prcp, weight), 2)

    blended_pred['Tmin'] = round(blend_temperature(api_data, pred_date, 'tmin', pred['Tmin'], day_weights['temp']), 2)
    blended_pred['Tmax'] = round(blend_temperature(api_data, pred_date, 'tmax', pred['Tmax'], day_weights['temp']), 2)

    blended_pred.update(get_weather_conditions(api_data, pred_date))

    add_rain_predictions(blended_pred, api_data, pred_date)

    return blended_pred

def blend_precipitation(overall_prob, api_prcp, model_prcp, weight):
    """Blend precipitation values"""
    if overall_prob < 0.20:
        return 0.00
    return max(0, (weight * api_prcp) + ((1 - weight) * model_prcp))

def blend_day_precipitation(day_prob, api_data, pred_date, model_prcp, weight):
    """Blend day precipitation values"""
    if day_prob < 0.20:
        return 0.00
    api_day_prcp = api_data.loc[pred_date, 'day_prcp']
    model_day_prcp = model_prcp * 0.5
    return max(0, (weight * api_day_prcp) + ((1 - weight) * model_day_prcp))

def blend_night_precipitation(night_prob, api_data, pred_date, model_prcp, weight):
    """Blend night precipitation values"""
    if night_prob < 0.20:
        return 0.00
    api_night_prcp = api_data.loc[pred_date, 'night_prcp']
    model_night_prcp = model_prcp * 0.5
    return max(0, (weight * api_night_prcp) + ((1 - weight) * model_night_prcp))

def blend_temperature(api_data, pred_date, temp_type, model_temp, weight):
    """Blend temperature values"""
    api_temp = api_data.loc[pred_date, temp_type]
    return (weight * api_temp) + ((1 - weight) * model_temp)

def get_weather_conditions(api_data, pred_date):
    """Get weather conditions from API data"""
    try:
        day_weather = api_data.loc[pred_date, 'day_weather']
        night_weather = api_data.loc[pred_date, 'night_weather']
        return {
            'Day Weather': day_weather,
            'Night Weather': night_weather
        }
    except Exception as e:
        print(f"Error getting weather conditions: {str(e)}")
        return {
            'Day Weather': "Unknown",
            'Night Weather': "Unknown"
        }

def add_rain_predictions(blended_pred, api_data, pred_date):
    """Add rain predictions to blended prediction"""
    day_precip_type = api_data.loc[pred_date, 'day_precip_type']
    day_precip_intensity = api_data.loc[pred_date, 'day_precip_intensity']
    day_precip_probability = round(api_data.loc[pred_date, 'day_precip_probability'], 2)
    night_precip_type = api_data.loc[pred_date, 'night_precip_type']
    night_precip_intensity = api_data.loc[pred_date, 'night_precip_intensity']
    night_precip_probability = round(api_data.loc[pred_date, 'night_precip_probability'], 2)

    day_rain_category, day_rain_intensity, _ = get_enhanced_rain_predictions(
        blended_pred['Day Precipitation (mm)'],
        day_precip_type,
        day_precip_intensity,
        day_precip_probability
    )

    night_rain_category, night_rain_intensity, _ = get_enhanced_rain_predictions(
        blended_pred['Night Precipitation (mm)'],
        night_precip_type,
        night_precip_intensity,
        night_precip_probability
    )

    blended_pred.update({
        'Day Rain Probability': day_precip_probability,
        'Night Rain Probability': night_precip_probability,
        'Day Rain Category': day_rain_category,
        'Day Rain Intensity': day_rain_intensity,
        'Day Precipitation Type': day_precip_type if day_precip_type != "None" else "None",
        'Night Rain Category': night_rain_category,
        'Night Rain Intensity': night_rain_intensity,
        'Night Precipitation Type': night_precip_type if night_precip_type != "None" else "None"
    })

    overall_prob = max(day_precip_probability, night_precip_probability)
    overall_rain_category, overall_rain_intensity, _ = get_enhanced_rain_predictions(
        blended_pred['Precipitation (mm)'],
        "Rain" if day_precip_type != "None" or night_precip_type != "None" else "None",
        day_precip_intensity if day_precip_probability > night_precip_probability else night_precip_intensity,
        overall_prob
    )

    blended_pred.update({
        'Rain Category': overall_rain_category,
        'Rain Probability': round(overall_prob, 2),
        'Rain Intensity': overall_rain_intensity
    })

    blended_pred['Precipitation Type'] = determine_precipitation_type(
        blended_pred['Day Weather'], blended_pred['Night Weather'],
        day_precip_type, night_precip_type
    )

def determine_precipitation_type(day_weather, night_weather, day_precip_type, night_precip_type):
    """Determine the overall precipitation type"""
    if "snow" in day_weather.lower() or "snow" in night_weather.lower() or "flurries" in day_weather.lower() or "flurries" in night_weather.lower():
        return "Snow"
    elif "sleet" in day_weather.lower() or "sleet" in night_weather.lower() or "freezing rain" in day_weather.lower() or "freezing rain" in night_weather.lower():
        return "Sleet"
    elif "hail" in day_weather.lower() or "hail" in night_weather.lower():
        return "Hail"
    elif "thunderstorm" in day_weather.lower() or "thunderstorm" in night_weather.lower() or "thunder" in day_weather.lower() or "thunder" in night_weather.lower():
        return "Thunderstorm"
    elif day_precip_type != "None" or night_precip_type != "None":
        return "Rain"
    else:
        return "None"

def create_fallback_prediction(pred):
    """Create a fallback prediction when API data is not available"""
    rain_category, rain_intensity, rain_prob = get_enhanced_rain_predictions(pred['Precipitation (mm)'])

    if rain_prob < 0.20:
        pred['Precipitation (mm)'] = 0.00

    if rain_prob < 0.20:
        weather = "Sunny"
    elif rain_prob < 0.50:
        weather = "Partly Cloudy"
    else:
        weather = "Rain"

    pred.update({
        'Rain Category': rain_category,
        'Rain Intensity': rain_intensity,
        'Rain Probability': rain_prob,
        'Precipitation Type': "Rain" if "High Chance" in rain_category else "None",
        'Day Precipitation (mm)': pred['Precipitation (mm)'] * 0.5,
        'Night Precipitation (mm)': pred['Precipitation (mm)'] * 0.5,
        'Day Rain Category': rain_category,
        'Day Rain Probability': rain_prob,
        'Day Rain Intensity': rain_intensity,
        'Day Precipitation Type': "Rain" if "High Chance" in rain_category else "None",
        'Night Rain Category': rain_category,
        'Night Rain Probability': rain_prob,
        'Night Rain Intensity': rain_intensity,
        'Night Precipitation Type': "Rain" if "High Chance" in rain_category else "None",
        'Day Weather': weather,
        'Night Weather': weather
    })

    return pred

# Retrieve it on AccuWeather API
def get_weather_predictions(api_key="need_gumawa_new_account", location_key="placeholder_location_key"): # Search kayo sa youtube on how to use AccuWeather API
    """Main function to get weather predictions"""
    try:
        data = load_models_and_data()
        if 'error' in data:
            return {'error': data['error']}

        accuweather_data = fetch_accuweather_data(api_key, location_key)
        
        if accuweather_data is None:
            return {'error': 'Failed to fetch AccuWeather data'}

        api_dates = accuweather_data.index.tolist()
        print(f"AccuWeather dates: {api_dates}")
        
        predictions = make_lstm_predictions(
            data['weather_scaled'],
            data['models'],
            data['scaler'],
            data['column_mapping'],
            api_dates=api_dates
        )

        #Blend predictions with AccuWeather data (can be changed if needed)
        blend_weights = {
            0: {'prcp': 0.90, 'temp': 0.90},
            1: {'prcp': 0.90, 'temp': 0.90},
            2: {'prcp': 0.85, 'temp': 0.85},
            3: {'prcp': 0.80, 'temp': 0.80},
            4: {'prcp': 0.80, 'temp': 0.80},
            'default': {'prcp': 0.0, 'temp': 0.0}
        }

        blended_predictions = blend_predictions(accuweather_data, predictions, blend_weights)

        display_predictions = []
        for pred in blended_predictions:
            pred_date = pred['Date'] if isinstance(pred['Date'], pd.Timestamp) else pd.Timestamp(pred['Date'])
            
            accu_data = accuweather_data.loc[pred_date]
            
            filtered_pred = {
                "Date": pred_date.strftime("%A, %B %d, %Y"),
                "Tmin": pred['Tmin'],
                "Tmax": pred['Tmax'],
                "Day Precipitation (mm)": pred.get('Day Precipitation (mm)', 0),
                "Night Precipitation (mm)": pred.get('Night Precipitation (mm)', 0),
                "Day Rain Probability": pred.get('Day Rain Probability', 0),
                "Night Rain Probability": pred.get('Night Rain Probability', 0),
                "Day Weather": accu_data['day_weather'],  
                "Night Weather": accu_data['night_weather']  
            }
            display_predictions.append(filtered_pred)

        blended_df = pd.DataFrame(blended_predictions)
        blended_df.set_index('Date', inplace=True)
        csv_path = os.path.join(predictions_dir, "weather_predictions.csv")
        blended_df.to_csv(csv_path)
        print(f"Weather predictions saved to: {csv_path}")

        return display_predictions

    except Exception as e:
        return {'error': f'Error generating predictions: {str(e)}'}

if __name__ == "__main__":
    result = get_weather_predictions()
    print(json.dumps(result, indent=2, default=str))