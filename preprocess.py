import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
RAW_PATH = r"C:\Users\MANJUSHREE\Desktop\archive (2)\CMaps\csv"       # Folder where raw CSVs are stored
SAVE_PATH = "./processed_data"


os.makedirs(SAVE_PATH, exist_ok=True)

for ds in DATASETS:
    print(f"\n--- Preprocessing {ds} ---")

    try:
       
        train = pd.read_csv(f"{RAW_PATH}/train_{ds}.csv")
        test = pd.read_csv(f"{RAW_PATH}/test_{ds}.csv")
        rul = pd.read_csv(f"{RAW_PATH}/RUL_{ds}.csv")

        train = train.dropna().reset_index(drop=True)

        max_cycle = train.groupby("unit_number")["time_in_cycles"].transform("max")
        train["RUL"] = max_cycle - train["time_in_cycles"]

        sensor_cols = [c for c in train.columns if "sensor" in c]

        scaler = MinMaxScaler()
        train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
        test[sensor_cols] = scaler.transform(test[sensor_cols])  # same scaler!

        train.to_csv(f"{SAVE_PATH}/train_{ds}_preprocessed.csv", index=False)
        test.to_csv(f"{SAVE_PATH}/test_{ds}_preprocessed.csv", index=False)
        rul.to_csv(f"{SAVE_PATH}/RUL_{ds}.csv", index=False)

        print(f" {ds} preprocessed and saved to {SAVE_PATH}")

    except Exception as e:
        print(f" Error processing {ds}: {e}")

