#!/bin/env python3
import sys
import nanoid
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check if CSV file argument is provided
if len(sys.argv) < 2:
    print("Error: Please provide the CSV file path as the first argument.")
    sys.exit(1)

# Load CSV file
csv_file = sys.argv[1]
data = pd.read_csv(csv_file)

# Combine latitude and longitude into a single target column
data['target_column'] = data[['latitude', 'longitude']].apply(tuple, axis=1)

# Split data into features and target
X = data.dropna().drop(columns=['target_column'])  # Features
y = data.dropna()['target_column']  # Combined latitude/longitude as target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train a small neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)  # Output layer with 2 units for latitude and longitude
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# here is where we save it to a file, with a uuid attached to the filename.
csv_file_stem = Path(csv_file).stem
model.save(f"{csv_file_stem}_{nanoid.generate()}")
