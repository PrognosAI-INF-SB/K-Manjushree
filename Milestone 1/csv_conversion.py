import os
import pandas as pd

data_dir = r"C:\Users\MANJUSHREE\Desktop\archive (2)\CMaps"


csv_dir = os.path.join(data_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)

columns = [
    "unit_number", "time_in_cycles",
    "operational_setting_1", "operational_setting_2", "operational_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]  # 21 sensors

def convert_to_csv(input_file, output_file, with_columns=True):
    print(f"🔹 Reading: {input_file}")
    df = pd.read_csv(input_file, sep=r"\s+", header=None)
    if with_columns:
        df.columns = columns[:df.shape[1]]
    df.to_csv(output_file, index=False)
    print(f" Saved: {output_file}")


for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)

    if file.startswith(("train_FD", "test_FD")):
        csv_path = os.path.join(csv_dir, file + ".csv")
        convert_to_csv(file_path, csv_path, with_columns=True)

    elif file.startswith("RUL_FD"):
        csv_path = os.path.join(csv_dir, file + ".csv")
        convert_to_csv(file_path, csv_path, with_columns=False)

print(f"\n All CSV files are saved in: {csv_dir}")
