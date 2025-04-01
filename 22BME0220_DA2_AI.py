# Import necessary libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Simulated dataset with more features for training
# Features: Temperature, Humidity, Smoke Level, Air Quality, Pressure, Sound Level (decibels)
X = np.array([
    [25, 40, 10, 85, 1013, 30],   # Normal conditions
    [30, 45, 20, 80, 1012, 35],   # Low risk
    [35, 50, 40, 75, 1011, 45],   # Moderate risk
    [40, 55, 60, 70, 1010, 55],   # High risk
    [45, 60, 80, 65, 1009, 65],   # High risk
    [50, 65, 90, 60, 1008, 75],   # Critical risk
    [55, 70, 110, 55, 1007, 85],  # Critical risk
    [60, 75, 120, 50, 1006, 95],  # Fire risk
    [65, 80, 130, 45, 1005, 105], # Fire risk
    [70, 85, 150, 40, 1004, 110]  # Fire risk
])

# Labels (0 = no fire risk, 1 = fire risk)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Standardize the features for deep learning
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),   # Hidden layer 1
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer 2
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Simulate sprinkler activation based on prediction
def activate_sprinkler(fire_risk_prediction, prediction_prob):
    if fire_risk_prediction == 1 and prediction_prob >= 0.7:
        print("Fire risk detected! Sprinkler system activated.")
    else:
        print("No fire risk detected. Sprinkler system is off.")

# Test cases with more complex environmental conditions
test_cases = [
    {"temp": 25, "humidity": 40, "smoke": 10, "air_quality": 85, "pressure": 1013, "sound": 30},  # Normal conditions
    {"temp": 50, "humidity": 65, "smoke": 90, "air_quality": 60, "pressure": 1008, "sound": 75},  # High risk
    {"temp": 60, "humidity": 75, "smoke": 120, "air_quality": 45, "pressure": 1005, "sound": 95}, # Fire risk
    {"temp": 70, "humidity": 85, "smoke": 150, "air_quality": 40, "pressure": 1004, "sound": 110}, # Fire risk
    {"temp": 30, "humidity": 45, "smoke": 20, "air_quality": 80, "pressure": 1012, "sound": 35},  # Low risk
]

# Evaluate each test case and trigger sprinkler based on predictions
for idx, case in enumerate(test_cases, 1):
    print(f"\nTest Case {idx}:")
    new_conditions = np.array([[case["temp"], case["humidity"], case["smoke"], case["air_quality"], case["pressure"], case["sound"]]])
    new_conditions_scaled = scaler.transform(new_conditions)
    
    # Predict fire risk
    fire_risk_prediction = model.predict(new_conditions_scaled)
    prediction_prob = fire_risk_prediction[0][0]
    
    print(f"Input Data: {case}")
    print(f"Prediction: {'Fire Risk' if prediction_prob >= 0.5 else 'No Fire Risk'}")
    print(f"Prediction Probability: {prediction_prob:.2f}")
    
    # Trigger sprinkler if fire risk is predicted
    activate_sprinkler(1 if prediction_prob >= 0.5 else 0, prediction_prob)
