#!/usr/bin/env python
# AI Prediction Engine Module

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PredictionEngine")

class ModelFactory:
    """Factory for creating various prediction models"""
    
    @staticmethod
    def create_model(model_type, input_shape=None, **kwargs):
        """
        Create a model based on specified type
        
        Args:
            model_type (str): Type of model to create
            input_shape (tuple): Shape of input data
            **kwargs: Additional parameters for model creation
            
        Returns:
            Model: The created model
        """
        if model_type == "lstm":
            return ModelFactory._create_lstm_model(input_shape, **kwargs)
        elif model_type == "gru":
            return ModelFactory._create_gru_model(input_shape, **kwargs)
        elif model_type == "bidirectional_lstm":
            return ModelFactory._create_bidirectional_lstm(input_shape, **kwargs)
        elif model_type == "cnn_lstm":
            return ModelFactory._create_cnn_lstm_model(input_shape, **kwargs)
        elif model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=42
            )
        elif model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 6),
                random_state=42
            )
        elif model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 6),
                random_state=42
            )
        elif model_type == "svr":
            return SVR(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                epsilon=kwargs.get("epsilon", 0.1)
            )
        elif model_type == "linear":
            return LinearRegression()
        elif model_type == "ridge":
            return Ridge(
                alpha=kwargs.get("alpha", 1.0),
                random_state=42
            )
        elif model_type == "ensemble":
            return ModelFactory._create_ensemble_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def _create_lstm_model(input_shape, **kwargs):
        """Create an LSTM model for time series prediction"""
        units = kwargs.get("units", [64, 32])
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        
        model = Sequential()
        model.add(LSTM(units[0], return_sequences=True if len(units) > 1 else False, 
                       input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(LSTM(units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "mean_squared_error")
        )
        
        return model
    
    @staticmethod
    def _create_gru_model(input_shape, **kwargs):
        """Create a GRU model for time series prediction"""
        units = kwargs.get("units", [64, 32])
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        
        model = Sequential()
        model.add(GRU(units[0], return_sequences=True if len(units) > 1 else False, 
                     input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(GRU(units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "mean_squared_error")
        )
        
        return model
    
    @staticmethod
    def _create_bidirectional_lstm(input_shape, **kwargs):
        """Create a Bidirectional LSTM model"""
        units = kwargs.get("units", [64, 32])
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        
        model = Sequential()
        model.add(Bidirectional(LSTM(units[0], return_sequences=True if len(units) > 1 else False), 
                               input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(Bidirectional(LSTM(units[i], return_sequences=return_sequences)))
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "mean_squared_error")
        )
        
        return model
    
    @staticmethod
    def _create_cnn_lstm_model(input_shape, **kwargs):
        """Create a hybrid CNN-LSTM model"""
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed
        
        # Parameters
        cnn_filters = kwargs.get("cnn_filters", [64, 128])
        cnn_kernel_size = kwargs.get("cnn_kernel_size", 3)
        lstm_units = kwargs.get("lstm_units", 50)
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        
        # Define input
        inputs = Input(shape=input_shape)
        
        # CNN layers
        x = Conv1D(filters=cnn_filters[0], kernel_size=cnn_kernel_size, 
                  activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        
        for filter_size in cnn_filters[1:]:
            x = Conv1D(filters=filter_size, kernel_size=cnn_kernel_size, 
                      activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM layer
        x = LSTM(lstm_units, return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=kwargs.get("learning_rate", 0.001)),
            loss=kwargs.get("loss", "mean_squared_error")
        )
        
        return model
    
    @staticmethod
    def _create_ensemble_model(**kwargs):
        """
        Create an ensemble of models
        Note: This is a meta-model concept, not a single Keras model
        """
        base_models = kwargs.get("base_models", ["random_forest", "xgboost", "lstm"])
        ensemble = {
            "type": "ensemble",
            "models": base_models,
            "weights": kwargs.get("weights", [1/len(base_models)] * len(base_models))
        }
        return ensemble


class PredictionEngine:
    """AI engine for stock market prediction"""
    
    def __init__(self, model_dir="models"):
        """
        Initialize the prediction engine
        
        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def train_model(self, symbol, X_train, y_train, X_test, y_test, model_type="lstm", **kwargs):
        """
        Train a model on the provided data
        
        Args:
            symbol (str): Stock symbol
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            model_type (str): Type of model to train
            **kwargs: Additional parameters for model creation
            
        Returns:
            tuple: (model, history, evaluation_metrics)
        """
        try:
            logger.info(f"Training {model_type} model for {symbol}")
            
            # Scale the target values for neural networks
            if model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                # Reshape y data for scaling
                y_scaler = MinMaxScaler()
                y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
                y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
                
                # Store the scaler for later use
                self.scalers[f"{symbol}_{model_type}_y"] = y_scaler
                
                # Create model
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = ModelFactory.create_model(model_type, input_shape, **kwargs)
                
                # Setup callbacks for early stopping and model checkpointing
                model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.h5")
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
                ]
                
                # Train the model
                history = model.fit(
                    X_train, y_train_scaled,
                    epochs=kwargs.get("epochs", 100),
                    batch_size=kwargs.get("batch_size", 32),
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate the model
                y_pred_scaled = model.predict(X_test)
                y_pred = y_scaler.inverse_transform(y_pred_scaled)
                
                # Store the model
                self.models[f"{symbol}_{model_type}"] = model
                
            else:  # Traditional ML models
                # For ensemble model
                if model_type == "ensemble":
                    ensemble_config = ModelFactory.create_model(model_type, **kwargs)
                    models = {}
                    predictions = []
                    
                    # Train each base model in the ensemble
                    for base_model_type in ensemble_config["models"]:
                        base_model = self.train_model(
                            symbol, X_train, y_train, X_test, y_test, 
                            model_type=base_model_type, **kwargs
                        )[0]
                        
                        models[base_model_type] = base_model
                        
                        # Get predictions from this model
                        if base_model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                            # For neural network models, get scaled predictions
                            y_scaler = self.scalers[f"{symbol}_{base_model_type}_y"]
                            y_pred_scaled = base_model.predict(X_test)
                            y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
                        else:
                            # For traditional ML models
                            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                            y_pred = base_model.predict(X_test_reshaped)
                            
                        predictions.append(y_pred)
                    
                    # Combine predictions using ensemble weights
                    weighted_preds = np.zeros_like(predictions[0])
                    for i, pred in enumerate(predictions):
                        weighted_preds += pred * ensemble_config["weights"][i]
                        
                    # Store the ensemble
                    ensemble_config["base_models"] = models
                    self.models[f"{symbol}_{model_type}"] = ensemble_config
                    
                    # Evaluate ensemble
                    y_pred = weighted_preds
                    model = ensemble_config
                    history = None
                    
                else:  # Standard ML models
                    # Reshape input for traditional ML models
                    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
                    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                    
                    # Create and train the model
                    model = ModelFactory.create_model(model_type, **kwargs)
                    model.fit(X_train_reshaped, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_reshaped)
                    
                    # Store the model
                    model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                        
                    self.models[f"{symbol}_{model_type}"] = model
                    history = None
            
            # Calculate evaluation metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            logger.info(f"Model evaluation for {symbol} {model_type}:")
            logger.info(f"RMSE: {metrics['rmse']:.6f}")
            logger.info(f"MAE: {metrics['mae']:.6f}")
            logger.info(f"R^2: {metrics['r2']:.6f}")
            
            # Save scaler
            if model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                scaler_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_y_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(y_scaler, f)
            
            return model, history, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None, None, None
    
    def predict(self, symbol, features, model_type="lstm"):
        """
        Make predictions using a trained model
        
        Args:
            symbol (str): Stock symbol
            features (numpy.ndarray): Input features for prediction
            model_type (str): Type of model to use for prediction
            
        Returns:
            float: Predicted value
        """
        try:
            model_key = f"{symbol}_{model_type}"
            
            if model_key not in self.models:
                # Try to load the model from disk
                self.load_model(symbol, model_type)
                
            if model_key not in self.models:
                logger.error(f"No trained model found for {symbol} with type {model_type}")
                return None
                
            model = self.models[model_key]
            
            # Handle ensemble model differently
            if isinstance(model, dict) and model.get("type") == "ensemble":
                predictions = []
                for base_model_type in model["models"]:
                    # Get prediction from base model
                    base_pred = self.predict(symbol, features, model_type=base_model_type)
                    if base_pred is not None:
                        predictions.append(base_pred)
                
                # Combine predictions using ensemble weights
                if len(predictions) > 0:
                    weighted_sum = sum(p * w for p, w in zip(predictions, model["weights"][:len(predictions)]))
                    return weighted_sum
                else:
                    return None
            
            # For deep learning models
            if model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                prediction = model.predict(features)
                
                # Inverse transform the prediction
                scaler_key = f"{symbol}_{model_type}_y"
                if scaler_key in self.scalers:
                    y_scaler = self.scalers[scaler_key]
                else:
                    # Try to load the scaler
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_y_scaler.pkl")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            y_scaler = pickle.load(f)
                            self.scalers[scaler_key] = y_scaler
                    else:
                        logger.error(f"No scaler found for {symbol} with model {model_type}")
                        return None
                
                prediction = y_scaler.inverse_transform(prediction)
                return prediction[0][0]
            else:
                # Reshape input for traditional ML models
                features_reshaped = features.reshape(features.shape[0], -1)
                prediction = model.predict(features_reshaped)
                return prediction[0]
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self, symbol, model_type="lstm"):
        """
        Save a trained model to disk
        
        Args:
            symbol (str): Stock symbol
            model_type (str): Type of model to save
        """
        try:
            model_key = f"{symbol}_{model_type}"
            
            if model_key not in self.models:
                logger.error(f"No model found for {symbol} with type {model_type}")
                return
                
            model = self.models[model_key]
            
            # Save ensemble model configuration
            if isinstance(model, dict) and model.get("type") == "ensemble":
                # Save the ensemble configuration
                config_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_config.pkl")
                ensemble_config = {
                    "type": "ensemble",
                    "models": model["models"],
                    "weights": model["weights"]
                }
                
                with open(config_path, 'wb') as f:
                    pickle.dump(ensemble_config, f)
                
                # Save each base model
                for base_model_type in model["models"]:
                    self.save_model(symbol, model_type=base_model_type)
                    
            elif model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                # Save Keras model
                model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.h5")
                model.save(model_path)
                
                # Save scaler
                scaler_key = f"{symbol}_{model_type}_y"
                if scaler_key in self.scalers:
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_y_scaler.pkl")
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[scaler_key], f)
            else:
                # Save traditional ML model
                model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
            logger.info(f"Saved model for {symbol} with type {model_type}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, symbol, model_type="lstm"):
        """
        Load a trained model from disk
        
        Args:
            symbol (str): Stock symbol
            model_type (str): Type of model to load
            
        Returns:
            object: The loaded model
        """
        try:
            model_key = f"{symbol}_{model_type}"
            
            # Check for ensemble model first
            config_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_config.pkl")
            if os.path.exists(config_path) and model_type == "ensemble":
                with open(config_path, 'rb') as f:
                    ensemble_config = pickle.load(f)
                
                # Load base models
                for base_model_type in ensemble_config["models"]:
                    self.load_model(symbol, model_type=base_model_type)
                
                # Store ensemble config
                self.models[model_key] = ensemble_config
                logger.info(f"Loaded ensemble model for {symbol}")
                return ensemble_config
                
            # For deep learning models
            elif model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.h5")
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    self.models[model_key] = model
                    
                    # Load scaler
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_y_scaler.pkl")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            y_scaler = pickle.load(f)
                            self.scalers[f"{symbol}_{model_type}_y"] = y_scaler
                    
                    logger.info(f"Loaded {model_type} model for {symbol}")
                    return model
                else:
                    logger.warning(f"No saved model found at {model_path}")
                    return None
            else:
                # For traditional ML models
                model_path = os.path.join(self.model_dir, f"{symbol}_{model_type}_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        self.models[model_key] = model
                        
                    logger.info(f"Loaded {model_type} model for {symbol}")
                    return model
                else:
                    logger.warning(f"No saved model found at {model_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def plot_performance(self, symbol, y_test, y_pred, model_type="lstm"):
        """
        Plot the model prediction performance
        
        Args:
            symbol (str): Stock symbol
            y_test (numpy.ndarray): Actual values
            y_pred (numpy.ndarray): Predicted values
            model_type (str): Type of model used
            
        Returns:
            str: Path to saved plot
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f"{symbol} Stock Price Prediction using {model_type.upper()}")
            plt.xlabel('Time')
            plt.ylabel('Price Change %')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plots_dir = os.path.join(self.model_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(plots_dir, f"{symbol}_{model_type}_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved performance plot to {plot_path}")
            return plot_path
            
        except Exception as e:
            logger.error(f"Error plotting performance: {str(e)}")
            return None
    
    def evaluate_all_models(self, symbol, X_test, y_test):
        """
        Evaluate all trained models for a given symbol
        
        Args:
            symbol (str): Stock symbol
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            
        Returns:
            dict: Evaluation metrics for each model
        """
        results = {}
        
        try:
            # Find all models for this symbol
            for model_key in list(self.models.keys()):
                if model_key.startswith(f"{symbol}_"):
                    model_type = model_key.replace(f"{symbol}_", "")
                    
                    # Make predictions
                    if model_type in ["lstm", "gru", "bidirectional_lstm", "cnn_lstm"]:
                        model = self.models[model_key]
                        y_pred_scaled = model.predict(X_test)
                        
                        # Inverse transform
                        scaler_key = f"{symbol}_{model_type}_y"
                        if scaler_key in self.scalers:
                            y_scaler = self.scalers[scaler_key]
                            y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
                        else:
                            logger.error(f"No scaler found for {model_key}")
                            continue
                    else:
                        # For traditional ML models or ensemble
                        if model_type == "ensemble":
                            # Use the predict method which handles ensembles
                            y_pred = np.array([self.predict(symbol, X_test[i:i+1], model_type) 
                                             for i in range(len(X_test))])
                        else:
                            # Reshape for traditional ML
                            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                            model = self.models[model_key]
                            y_pred = model.predict(X_test_reshaped)
                    
                    # Calculate metrics
                    metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    
                    results[model_type] = {
                        'metrics': metrics,
                        'predictions': y_pred
                    }
                    
                    logger.info(f"Evaluation for {model_key}:")
                    logger.info(f"RMSE: {metrics['rmse']:.6f}")
                    logger.info(f"MAE: {metrics['mae']:.6f}")
                    logger.info(f"R^2: {metrics['r2']:.6f}")
                    
            return results
                    
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            return results
    
    def get_best_model(self, symbol, evaluation_results=None, metric='rmse'):
        """
        Determine the best model for a symbol based on evaluation metrics
        
        Args:
            symbol (str): Stock symbol
            evaluation_results (dict): Evaluation results from evaluate_all_models
            metric (str): Metric to use for comparison ('rmse', 'mae', 'r2')
            
        Returns:
            str: The best model type
        """
        try:
            if evaluation_results is None:
                logger.error("No evaluation results provided")
                return None
                
            best_score = float('inf')  # For rmse and mae, lower is better
            reverse = False
            
            if metric == 'r2':  # For R^2, higher is better
                best_score = float('-inf')
                reverse = True
                
            best_model = None
            
            for model_type, results in evaluation_results.items():
                score = results['metrics'][metric]
                
                if reverse:
                    # For metrics where higher is better
                    if score > best_score:
                        best_score = score
                        best_model = model_type
                else:
                    # For metrics where lower is better
                    if score < best_score:
                        best_score = score
                        best_model = model_type
            
            logger.info(f"Best model for {symbol} based on {metric}: {best_model} ({best_score:.6f})")
            return best_model
            
        except Exception as e:
            logger.error(f"Error finding best model: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # This is a demonstration of how to use the PredictionEngine
    import sys
    sys.path.append('..')  # Add parent directory to path
    from data_collection import DataCollector
    
    # Initialize
    collector = DataCollector()
    engine = PredictionEngine()
    
    # Fetch and prepare data
    symbol = "AAPL"
    collector.fetch_stock_data(symbol, period="2y", interval="1d")
    X_train, y_train, X_test, y_test, data = collector.prepare_training_data(symbol)
    
    if X_train is not None:
        # Train LSTM model
        lstm_model, lstm_history, lstm_metrics = engine.train_model(
            symbol, X_train, y_train, X_test, y_test, 
            model_type="lstm",
            units=[64, 32],
            dropout_rate=0.2,
            epochs=50
        )
        
        # Train Random Forest model
        rf_model, _, rf_metrics = engine.train_model(
            symbol, X_train, y_train, X_test, y_test,
            model_type="random_forest",
            n_estimators=100,
            max_depth=10
        )
        
        # Train an ensemble model
        ensemble_model, _, ensemble_metrics = engine.train_model(
            symbol, X_train, y_train, X_test, y_test,
            model_type="ensemble",
            base_models=["lstm", "random_forest", "xgboost"],
            weights=[0.5, 0.3, 0.2]
        )
        
        # Evaluate all models
        evaluation = engine.evaluate_all_models(symbol, X_test, y_test)
        
        # Get the best model
        best_model = engine.get_best_model(symbol, evaluation, metric='rmse')
        
        # Get the latest data for prediction
        latest_features = collector.get_latest_features(symbol)
        
        if latest_features is not None:
            # Make prediction using best model
            prediction = engine.predict(symbol, latest_features, model_type=best_model)
            print(f"\nPredicted price change for {symbol} using {best_model}: {prediction:.4f}%")
