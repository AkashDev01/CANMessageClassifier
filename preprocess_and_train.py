import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump

def read_csv(file_path):
    return pd.read_csv(file_path)

def hex_to_decimal(df, hex_columns):
    for column in hex_columns:
        df[column] = df[column].apply(lambda x: int(x, 16) if pd.notnull(x) else x)
    return df

def encode_labels(df, label_column):
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])
    return df, label_encoder

def clean_data(df):
    # Drop rows with missing values & remove unnecessary columns if we need.
    return df

def extract_features_and_labels(df):
    feature_columns = ['Time Stamp', 'ID', 'LEN', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    label_column = 'PGN LABEL'
    X = df[feature_columns]
    y = df[label_column]
    return X, y

def normalize_features(X):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(file_path):
    df = read_csv(file_path)
    df = clean_data(df)
    df = hex_to_decimal(df, ['ID', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'PGN'])
    df, label_encoder = encode_labels(df, 'PGN LABEL')
    X, y = extract_features_and_labels(df)
    X = normalize_features(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    file_path = 'data.csv'
    X_train, X_test, y_train, y_test, label_encoder = preprocess(file_path)


    # Save the label encoder
    dump(label_encoder, 'label_encoder.joblib')

    # LSTM Model Training
    timesteps = 1  
    X_train = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stop])

    # Save the model
    model.save('lstm_can_model.h5')

