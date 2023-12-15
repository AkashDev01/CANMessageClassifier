import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

from sklearn.preprocessing import StandardScaler

def hex_to_decimal(df, hex_columns):
    for column in hex_columns:
        df[column] = df[column].apply(lambda x: int(x, 16) if pd.notnull(x) else x)
    return df

def normalize_features(X):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def preprocess_data(df):
    # Convert hexadecimal fields to decimal
    df = hex_to_decimal(df, ['ID', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'])

    # Select the relevant features
    # Note: Adjust the feature columns as per your model's training data
    feature_columns = ['Time Stamp', 'ID', 'LEN', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    X = df[feature_columns]

    # Normalize the features
    X = normalize_features(X)

    return X
def extract_pgn(can_id):
    # Convert the CAN ID from hex to a 29-bit binary string
    binary_id = format(int(str(can_id), 16), '029b')    
    # Extract PGN (18-25 and 8-17 bits in the 29-bit ID)
    pgn_binary = binary_id[1:9] + binary_id[9:19]
    
    # Convert the binary PGN to a decimal value
    pgn_decimal = int(pgn_binary, 2)
    return pgn_decimal

# Load the test dataset
test_df = pd.read_csv('test.csv')

# Preprocess the test data
X_test = preprocess_data(test_df)

# Reshape data for LSTM (if necessary)
timesteps = 1  # The same timesteps used during training
X_test = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# Load the trained model and label encoder
model = load_model('lstm_can_model.h5')
label_encoder = load('label_encoder.joblib')

# Predict using the model
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=-1)

# Decode the predicted labels
decoded_predictions = label_encoder.inverse_transform(predicted_labels)

# Extract PGN details
test_df['PGN'] = test_df['ID'].apply(extract_pgn)

# Combine predictions with the test data
test_df['Predicted PGN Label'] = decoded_predictions

# Save or display the results
test_df.to_csv('predicted_test_data.csv', index=False)
print(test_df)
