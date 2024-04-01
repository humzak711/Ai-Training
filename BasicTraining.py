import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os


# Process datasets to pandas dataframes
def process_csv_files(directory_path:str) -> list[str] | str:
    '''
    Iterate over given directory and process each dataset into
    a pandas dataframe then extract the data
    '''
    X_train = []
    Y_train = []
    feature_columns = None
    label_column = None
    columns_found = False

    # Iterate through each dataset and convert each csv to pandas dataframe
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)  # Read the CSV file into a Pandas DataFrame
            columns = df.columns.tolist()  # Extract column names from the DataFrame

            # Identify potential label and feature columns
            label_column_candidates = [col for col in columns if col != columns[-1]]
            if len(label_column_candidates) > 0:
                label_column = label_column_candidates[0]
                feature_columns = [col for col in columns if col != label_column]

            # Ensure the feature and label columns are found
            if feature_columns is not None and label_column is not None:
                columns_found = True

                # Handle non-numeric or string columns, converting them to numerical representation
                for col in feature_columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype('category').cat.codes  # Convert strings to numerical representation

                features = df[feature_columns].values
                labels = df[label_column].values

                X_train.append(features)
                Y_train.append(labels)  # Append the labels directly

            else:
                print(f"Warning: No suitable columns found in {filename}. Skipping this file.")

    if not columns_found:
        return None, None
    else:
        return np.vstack(X_train), np.hstack(Y_train)  # Use np.hstack to concatenate the label arrays

# Directory path containing CSV files
target_directory_path = 'Training_Data'
X_train, Y_train = process_csv_files(target_directory_path)

# Convert processed datasets to tensors
X_train = torch.tensor(X_train.astype(np.float32), dtype=torch.float32)  # Convert to float32

# Encode string labels to numerical representations
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)

# Convert encoded labels to PyTorch tensor
Y_train = torch.tensor(Y_train_encoded, dtype=torch.long)

# Split into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Convert encoded labels to PyTorch tensor
Y_train = torch.tensor(Y_train_encoded, dtype=torch.long)

# Get the number of unique classes for output_size
output_size = len(np.unique(Y_train.numpy()))  # Number of unique classes in training labels

# Initialize the model
input_size = len(X_train[0]) if len(X_train) > 0 else 0
model = NeuralNetwork(input_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Function to train the model
def TrainAI(model_file:str, epochs:int=1) -> str :

    ''' Trains the AI model within a designated number of epochs (default = 1)
        and saves it within a designated file'''

    # Check if the model file exists for loading the trained model
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        model.train()  # Set the model to training mode for further training
        print("Model loaded successfully. Continuing training.")
    else:
        print("No pre-trained model found. Starting training from scratch.")

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)

        # Resize the target labels to match the output size
        Y_train_resized = Y_train[:outputs.size(0)]  # Adjust the size of target labels

        loss = criterion(outputs, Y_train_resized)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Save the updated trained model
    torch.save(model.state_dict(), model_file)


# Start training
if __name__ == '__main__':
    TrainAI('model.pth', 100)
