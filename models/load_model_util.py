# load_model_util.py
import os
import pickle
import joblib

from tensorflow.keras.models import load_model

def load_model_and_scaler(symbol, model_type, model_dir="models"):
    """
    Load a trained model and optional scaler from the models directory.

    Parameters:
        symbol (str): e.g. 'RELIANCE'
        model_type (str): e.g. 'lstm', 'xgboost'
        model_dir (str): Base path where models are saved

    Returns:
        model, scaler (scaler may be None if not found)
    """
    model_name = f"{symbol}_{model_type}_model"

    model = None
    # Load model
    if model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        try:
            model = load_model(model_path)
            print(f"Loaded Keras model from {model_path}")
        except Exception as e:
            print(f"Failed to load Keras model from {model_path}: {e}")
            return None, None

    elif model_type in ["xgboost", "lightgbm", "random_forest", "svr", "ridge", "linear"]:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded ML model from {model_path}")
        except Exception as e:
            print(f"Failed to load ML model from {model_path}: {e}")
            return None, None

    else:
        raise ValueError("Unsupported model type")

    # Load scaler
    scaler_path = os.path.join(model_dir, "scalers", f"{symbol}_{model_type}_scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Scaler file exists but failed to load from {scaler_path}: {e}")
            scaler = None  # Explicitly reset if loading failed

    return model, scaler
