# save_model_util.py
import os
import joblib
import pickle

from tensorflow.keras.models import save_model

def save_model_and_scaler(symbol, model_type, model, scaler=None, model_dir="models"):
    """
    Saves the trained model and optionally the scaler to the models directory.

    Parameters:
        symbol (str): Stock symbol (e.g. 'RELIANCE')
        model_type (str): Type of model (e.g. 'lstm', 'xgboost')
        model: Trained model object
        scaler: Fitted scaler object (e.g. MinMaxScaler)
        model_dir (str): Base directory to store models
    """
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"{symbol}_{model_type}_model"
    
    # Save model
    if model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
        path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(path)
        print(f"Saved Keras model to {path}")
    elif model_type in ["xgboost", "lightgbm", "random_forest", "svr", "ridge", "linear"]:
        path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved ML model to {path}")

    # Save scaler if applicable
    if scaler:
        scaler_dir = os.path.join(model_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_dir, f"{symbol}_{model_type}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
