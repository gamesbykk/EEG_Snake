import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Add, MaxPooling1D, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model
warnings.filterwarnings('ignore')

class Muse2LSTMProcessor:
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
        # Muse 2 specific channel configuration
        self.channel_cols = ['TP9', 'AF7', 'AF8', 'TP10']
        self.n_channels = len(self.channel_cols)
        
        # Frequency band definitions (Hz) - adjusted for Muse 2
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
    def load_data(self, filepath):
        """Load EEG data from CSV file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df['action']=df['action'].replace("no_action", pd.NA).ffill()
        # Ensure we have the expected channels
        for ch in self.channel_cols:
            if ch not in df.columns:
                raise ValueError(f"Missing expected Muse 2 channel: {ch}")
        
        print(f"Data loaded: {len(df)} samples, {self.n_channels} channels")
        if 'action' in df.columns:
            print(f"Actions: {df['action'].value_counts().to_dict()}")
        
        return df
    
    def remove_artifacts(self, df, method='statistical'):
        """Remove artifacts from EEG data"""
        print("Removing artifacts...")
        df_clean = df.copy()
        
        # Method 1: Statistical outlier removal
        if method == 'statistical':
            for channel in self.channel_cols:
                # Use robust statistics
                median_val = df_clean[channel].median()
                iqr_val = df_clean[channel].quantile(0.75) - df_clean[channel].quantile(0.25)
                threshold = 2.5 * iqr_val  # Less aggressive than before
                
                # Mark outliers as NaN
                outlier_mask = np.abs(df_clean[channel] - median_val) > threshold
                df_clean.loc[outlier_mask, channel] = np.nan
        
        # Interpolate missing values with limit
        for channel in self.channel_cols:
            df_clean[channel] = df_clean[channel].interpolate(method='linear', limit=3)
        
        # Forward/backward fill any remaining NaNs
        df_clean = df_clean.ffill().bfill()
        
        print(f"Artifacts removed: {df[self.channel_cols].isna().sum().sum()} samples interpolated")
        return df_clean
    
    def apply_filters(self, df):
        """Apply frequency filtering optimized for Muse 2"""
        print("Applying frequency filters...")
        df_filtered = df.copy()
        
        # Notch filter for 50/60 Hz line noise
        for freq in [50, 60]:
            b_notch, a_notch = signal.iirnotch(freq, Q=30, fs=self.sampling_rate)
            for channel in self.channel_cols:
                df_filtered[channel] = signal.filtfilt(b_notch, a_notch, df_filtered[channel])
        
        # Bandpass filter (1-40 Hz)
        b_bp, a_bp = signal.butter(4, [1, 40], btype='bandpass', fs=self.sampling_rate)
        for channel in self.channel_cols:
            df_filtered[channel] = signal.filtfilt(b_bp, a_bp, df_filtered[channel])
        
        return df_filtered
    
    def create_sequences(self, data, labels, sequence_length=256):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(labels[i+sequence_length-1])  # Label at the end of sequence
        return np.array(X), np.array(y)
    
    def prepare_for_lstm(self, df, target_column='action', sequence_length=256):
        """Prepare data for LSTM training"""
        print(f"Preparing LSTM sequences with length {sequence_length}...")
        
        # Extract features and labels
        X = df[self.channel_cols].values
        y = df[target_column].values

        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Create sequences
        X_sequences, y_sequences = self.create_sequences(X_scaled, y_categorical, sequence_length)
        
        print(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        return X_sequences, y_sequences
    def load_model(self):
        model=self.build_lstm_model((256,4),4)
        model.load_weights('muse2_lstm_model.weights.h5')
        return model
    

    def tcn_block(self,x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
        """A residual TCN block with two dilated causal conv layers"""
        # First dilated causal conv
        out = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                    padding='causal')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(dropout_rate)(out)
        
        # Second dilated causal conv
        out = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                    padding='causal')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(dropout_rate)(out)
        
        # Residual connection (1x1 conv if needed for dimension match)
        res = Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
        return Add()([res, out])

    def build_lstm_model(self,input_shape, num_classes):
        inp = Input(shape=input_shape)

        # --- TCN branch ---
        tcn = inp
        for d in [1, 2, 4, 8]:  # exponentially increasing dilation rates
            tcn = self.tcn_block(tcn, filters=64, kernel_size=3, dilation_rate=d, dropout_rate=0.3)

                # --- CNN branch ---
        cnn = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inp)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.3)(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)

        # --- Merge TCN + CNN ---
        # Ensure same time dimension before merge (use Add if same, Concatenate if not)
        merged = Add()([tcn, cnn]) if tcn.shape[1] == cnn.shape[1] else tcn

        # --- LSTM layer ---
        lstm = LSTM(128, return_sequences=False)(merged)
        lstm = Dropout(0.3)(lstm)

        # --- Dense classifier ---
        dense = Dense(64, activation='relu')(lstm)
        dense = Dropout(0.2)(dense)
        out = Dense(num_classes, activation='softmax')(dense)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    
    def train_lstm_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
        """Train the LSTM model"""
        print("Training LSTM model...")
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        model = self.build_lstm_model(input_shape, num_classes)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(np.argmax(y_train, axis=1)),
            y=np.argmax(y_train, axis=1)
        )
        class_weights_dict = dict(enumerate(class_weights))
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1,
            class_weight=class_weights_dict
        )
        model.save_weights('muse2_lstm_model.weights.h5')
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                 target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    def preprocess_pipeline(self, filepath):
        """Complete preprocessing pipeline for LSTM"""
        print("Starting Muse 2 LSTM preprocessing pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        
        
        return df


def main():
    """Main function to demonstrate the LSTM pipeline"""
    # Initialize preprocessor
    processor = Muse2LSTMProcessor(sampling_rate=256)
    
    # Example usage
    input_file = "all_eeg_snake_data.csv"  # Replace with your file
    
    try:
        # Run preprocessing pipeline
        df_processed = processor.preprocess_pipeline(input_file)
        
        # Prepare LSTM sequences
        sequence_length = 256  # 1 second of data at 256Hz
        X, y = processor.prepare_for_lstm(df_processed, sequence_length=sequence_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = processor.train_lstm_model(X_train, y_train, X_test, y_test, epochs=100)
        
        # Evaluate
        processor.evaluate_model(model, X_test, y_test)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
