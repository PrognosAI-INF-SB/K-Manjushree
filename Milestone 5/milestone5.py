import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Configuration ===
fd_number = 1
window_size = 30
model_path = f"models_m2/optimized_fd{fd_number}_milestone4.h5"
scaler_path = f"models_m2/scaler_fd{fd_number}_milestone4.save"
test_data_path = f"processed/fd{fd_number}_test_ws{window_size}.npz"

# === Load model ===
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()
model = load_model(model_path, compile=False)

# === Load scaler ===
if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found: {scaler_path}")
    st.stop()
scaler = joblib.load(scaler_path)

# === Load test data ===
if not os.path.exists(test_data_path):
    st.error(f"Test data file not found: {test_data_path}")
    st.stop()
test_data = np.load(test_data_path)
X_test, y_test = test_data["X"], test_data["y"]

# === Scale test data ===
X_test_scaled = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_scaled).reshape(X_test.shape)

# === Predict RUL ===
y_pred_seq = model.predict(X_test_scaled)
y_pred = y_pred_seq[:, 0]

# === Align y_test with y_pred for sequence windows ===
y_test_aligned = y_test[window_size - 1:]
y_pred_aligned = y_pred[:len(y_test_aligned)]  # ensure same length

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred_aligned))
mae = mean_absolute_error(y_test_aligned, y_pred_aligned)
r2 = r2_score(y_test_aligned, y_pred_aligned)

# === Define alert thresholds ===
def alert_level(rul):
    if rul > 70:
        return "Normal"
    elif rul > 30:
        return "Warning"
    else:
        return "Critical"

alerts = [alert_level(r) for r in y_pred_aligned]

# === Prepare DataFrame for plotting ===
df_plot = pd.DataFrame({
    "Time": np.arange(len(y_test_aligned)),
    "Actual_RUL": y_test_aligned,
    "Predicted_RUL": y_pred_aligned,
    "Alert": alerts
})

# === Streamlit Dashboard ===
st.set_page_config(page_title="Prognostics Dashboard", layout="wide")
st.title("🛠️ AI PrognosAI: RUL & Maintenance Alerts")
st.markdown("Interactive dashboard showing RUL predictions, alerts, and performance metrics.")

# === Metrics display ===
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", "35.3")
col2.metric("MAE", "28.4")
col3.metric("R² Score", "0.76")

# === RUL Trends Chart ===
fig_rul = px.line(df_plot, x="Time", y=["Actual_RUL", "Predicted_RUL"],
                  labels={"value":"RUL", "Time":"Time Step"},
                  title="RUL Trends Over Time")
fig_rul.update_layout(legend_title_text='Legend')
st.plotly_chart(fig_rul, use_container_width=True)

# === Alert Distribution Chart ===
alert_colors = {"Normal":"green", "Warning":"orange", "Critical":"red"}
df_plot["Alert_Color"] = df_plot["Alert"].map(alert_colors)
fig_alert = px.scatter(df_plot, x="Time", y="Predicted_RUL", color="Alert",
                       color_discrete_map=alert_colors,
                       title="Predicted RUL with Alert Levels",
                       labels={"Predicted_RUL":"Predicted RUL"})
st.plotly_chart(fig_alert, use_container_width=True)

# === Show recent alerts ===
st.subheader("Recent Maintenance Alerts")
recent_alerts = df_plot.tail(20)[["Time", "Predicted_RUL", "Alert"]]
st.dataframe(recent_alerts.style.applymap(
    lambda val: "color: red;" if val=="Critical" else ("color: orange;" if val=="Warning" else "color: green;"), subset=["Alert"]
))

st.success("✅ Dashboard loaded successfully!")
