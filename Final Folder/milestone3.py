# milestone3.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Configuration ===
fd_number = 1
window_size = 30  # must match your preprocessed data
data_dir = "processed"
model_path = f"models_m2/optimized_fd{fd_number}.h5"

# === Create folders for graphs and results ===
os.makedirs("graphs", exist_ok=True)
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# === Load test data and trained model ===
print("🔹 Loading test data and trained model...")
test_file = f"{data_dir}/fd{fd_number}_test_ws{window_size}.npz"
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file not found: {test_file}")

test_data = np.load(test_file)
X_test, y_test = test_data["X"], test_data["y"]

# Load model (handle legacy mse metric if needed)
model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

# === Predict Remaining Useful Life (RUL) ===
print("🔹 Making predictions on test data...")
y_pred = model.predict(X_test).flatten()

# === Compute performance metrics ===
print("🔹 Calculating performance metrics...")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 100 - (rmse / np.max(y_test) * 100)

# === Save metrics to CSV ===
results = pd.DataFrame({
    "FD_Set": [f"FD00{fd_number}"],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2_Score": [r2],
    "Accuracy(%)": [accuracy]
})
results.to_csv(os.path.join(results_folder, "model_performance.csv"), index=False)
print(f"📊 Model performance saved to {results_folder}/model_performance.csv")

# === Print metrics ===
print(f"\nModel Evaluation Report — FD00{fd_number}")
print("===========================================")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R² Score : {r2:.4f}")
print(f"Approx. Accuracy : {accuracy:.2f}%")

# === Plot 1: Predicted vs Actual RUL ===
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.5, color="dodgerblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title(f"Predicted vs Actual RUL (FD00{fd_number})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("graphs", f"fd{fd_number}_predicted_vs_actual.png"))
plt.close()

# === Plot 2: Residual Distribution ===
residuals = y_test - y_pred
plt.figure(figsize=(7,5))
sns.histplot(residuals, bins=40, kde=True, color="mediumvioletred")
plt.title(f"Residual Distribution (FD00{fd_number})")
plt.xlabel("Prediction Error (y_test - y_pred)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join("graphs", f"fd{fd_number}_residual_distribution.png"))
plt.close()

# === Plot 3: Residual Trend ===
plt.figure(figsize=(10,5))
plt.plot(residuals[:600], color="orange")
plt.title(f"Residual Trend (First 600 Samples) — FD00{fd_number}")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Error)")
plt.tight_layout()
plt.savefig(os.path.join("graphs", f"fd{fd_number}_residual_trend.png"))
plt.close()

# === Model Bias & Error Analysis ===
print("\n🔍 Performing bias and error analysis...")
mean_res = np.mean(residuals)
std_res = np.std(residuals)
bias_type = "Underestimation" if mean_res > 0 else "Overestimation"
bias_strength = "Low" if abs(mean_res) < std_res * 0.5 else "High"

bias_report = {
    "Mean_Residual": [mean_res],
    "Std_Residual": [std_res],
    "Bias_Type": [bias_type],
    "Bias_Strength": [bias_strength]
}
bias_df = pd.DataFrame(bias_report)
bias_df.to_csv(os.path.join(results_folder, "model_bias_analysis.csv"), index=False)
print(f"📘 Model Bias Type: {bias_type} ({bias_strength})")
print(f"✅ Bias analysis saved to {results_folder}/model_bias_analysis.csv")

print("\n📈 Evaluation graphs saved in ./graphs/")
print("\n🎯 Milestone 3 completed successfully.")
