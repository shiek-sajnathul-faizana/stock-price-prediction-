import pandas as pd
df = pd.read_csv('Modified_msft.csv')
print(df)

#feature engineering
#--adding lag values
df['Volume_lag1']=df['Volume'].shift(1)
df['Volume_lag2']=df['Volume'].shift(2)

#-- 7 day moving average
df['7-day_MA'] = df['Volume'].rolling(window=7).mean()
print(df)

# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load your time series data or sequence data
# Replace 'data' with your time series data
# Example: data = load_your_data()

# Data preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Volume']])

# Define the sequence length and split the data into sequences
sequence_length = 10  # Adjust this based on your problem
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to match the expected input shape
num_features = X_train.shape[2]  # Extract the number of features from the reshaped data

# Define and compile the LSTM model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, activation='relu', input_shape=(sequence_length, num_features)))
model.add(keras.layers.Dense(1))  # Adjust the output layer for your problem
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform predictions to the original scale if needed
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Evaluate the model using appropriate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print or visualize the model's performance metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R²): {r2}")
