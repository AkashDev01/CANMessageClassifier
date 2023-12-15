# CAN Message Classifier

This project includes a set of scripts to preprocess and classify CAN messages using an LSTM neural network model.

## Description

CAN Message Classifier is designed to process messages from the CAN bus and classify them based on historical data using machine learning techniques.

## Getting Started

### Dependencies

Required libraries can be found in requirements.txt

### Executing program

 1. Clone the repo.
 2. Create a virtual environment for running the scripts.
 3. pip install -r requirements.txt , data.csv contains the train, test data.
 4. python ./preprocess_and_train.py
 5. python ./main.py . test.csv contains new unseen data during training, 
    use our trained model to predict and it willbe saved as predicted_test_data.csv.


