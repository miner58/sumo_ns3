import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

class StackedLSTMModel(pl.LightningModule):
    """
    Stacked LSTM model for traffic flow prediction.
    Uses a 4-layer LSTM with 128 neurons per layer and dropout of 0.2.
    """
    
    def __init__(self, 
                input_size: int = 1, 
                hidden_size: int = 128, 
                num_layers: int = 4,
                output_size: int = 1,
                dropout: float = 0.2,
                learning_rate: float = 1e-3):
        """
        Initialize the Stacked LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in each LSTM layer
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # For logging
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Final fully connected layer
        output = self.fc(lstm_out)
        
        return output
        
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Tuple of (x, y)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Tuple of (x, y)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Tuple of (x, y)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Log loss
        self.log('test_loss', loss)
        
        return loss
        
    def configure_optimizers(self):
        """
        Configure the optimizer.
        
        Returns:
            Adam optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class TrafficDataset(torch.utils.data.Dataset):
    """
    Dataset for traffic flow prediction.
    Converts time series data into sequences for LSTM training.
    """
    
    def __init__(self, 
                 data: np.ndarray, 
                 sequence_length: int = 12,
                 prediction_horizon: int = 1):
        """
        Initialize the traffic dataset.
        
        Args:
            data: Input data array with traffic flow values
            sequence_length: Length of input sequence
            prediction_horizon: How many steps ahead to predict
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
        
    def __getitem__(self, idx):
        """
        Get a sequence and its corresponding target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence, target)
        """
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length + self.prediction_horizon - 1]
        
        return sequence, target


class TrafficPredictor:
    """
    Traffic flow predictor using a Stacked LSTM model.
    Handles data preprocessing, model training, and inference.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 sequence_length: int = 12,
                 prediction_horizon: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3):
        """
        Initialize the traffic predictor.
        
        Args:
            model_path: Path to saved model (if None, a new model is created)
            sequence_length: Length of input sequence for prediction
            prediction_horizon: How many steps ahead to predict
            hidden_size: Number of hidden units in each LSTM layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            self.model = StackedLSTMModel.load_from_checkpoint(model_path)
            print(f"Model loaded from: {model_path}")
        else:
            self.model = StackedLSTMModel(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=dropout,
                learning_rate=learning_rate
            )
            print("Created new model")
            
        # Data scaling
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def preprocess_data(self, data: pd.DataFrame, column: str) -> np.ndarray:
        """
        Preprocess the data for training or inference.
        
        Args:
            data: DataFrame containing time series data
            column: Name of the column to use for prediction
            
        Returns:
            Scaled numpy array
        """
        # Extract the target column
        values = data[column].values.reshape(-1, 1)
        
        # Fit and transform
        scaled_values = self.scaler.fit_transform(values)
        
        return scaled_values
        
    def create_datasets(self, 
                       scaled_data: np.ndarray, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       batch_size: int = 32):
        """
        Create training, validation, and test datasets.
        
        Args:
            scaled_data: Scaled input data
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            batch_size: Batch size for dataloaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split data into train, validation, and test sets
        n = len(scaled_data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = scaled_data[:train_size]
        val_data = scaled_data[train_size:train_size + val_size]
        test_data = scaled_data[train_size + val_size:]
        
        # Create datasets
        train_dataset = TrafficDataset(train_data, self.sequence_length, self.prediction_horizon)
        val_dataset = TrafficDataset(val_data, self.sequence_length, self.prediction_horizon)
        test_dataset = TrafficDataset(test_data, self.sequence_length, self.prediction_horizon)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )
        
        return train_loader, val_loader, test_loader
        
    def train(self, 
             train_loader, 
             val_loader,
             epochs: int = 600,
             gpus: int = 0,
             save_path: Optional[str] = None):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            gpus: Number of GPUs to use (0 for CPU)
            save_path: Path to save the trained model
            
        Returns:
            Trained model
        """
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='gpu' if gpus > 0 else 'cpu',
            devices=gpus if gpus > 0 else None,
            log_every_n_steps=10,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    filename='{epoch}-{val_loss:.4f}'
                ),
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=30,
                    mode='min'
                )
            ]
        )
        
        # Train model
        trainer.fit(self.model, train_loader, val_loader)
        
        # Save model
        if save_path:
            trainer.save_checkpoint(save_path)
            print(f"Model saved to: {save_path}")
            
        return self.model
        
    def predict(self, sequence: np.ndarray) -> float:
        """
        Make a prediction based on a sequence of traffic flow values.
        
        Args:
            sequence: Input sequence of shape (sequence_length, 1)
            
        Returns:
            Predicted traffic flow value
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Check sequence length
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Input sequence must have length {self.sequence_length}")
            
        # Scale sequence if it's not already scaled
        if sequence.max() > 1.0 or sequence.min() < 0.0:
            sequence = self.scaler.transform(sequence.reshape(-1, 1))
            
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            
        # Inverse transform to get original scale
        prediction_np = prediction.numpy().reshape(-1, 1)
        prediction_orig = self.scaler.inverse_transform(prediction_np)
        
        return float(prediction_orig[0, 0])
        
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")
        
    def load_model(self, path: str):
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from: {path}")
        
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create trainer for testing
        trainer = pl.Trainer(
            accelerator='cpu',
            devices=None
        )
        
        # Test model
        test_results = trainer.test(self.model, test_loader)
        
        return test_results[0] 