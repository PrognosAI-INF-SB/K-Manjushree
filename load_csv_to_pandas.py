import pandas as pd
import os

# Path to CMaps folder
cmap_folder = r"C:\Users\MANJUSHREE\Desktop\archive (2)\CMaps"

column_names = ['unit_number','time_in_cycles',
                'operational_setting_1','operational_setting_2','operational_setting_3',
                'sensor_1','sensor_2','sensor_3','sensor_4','sensor_5',
                'sensor_6','sensor_7','sensor_8','sensor_9','sensor_10',
                'sensor_11','sensor_12','sensor_13','sensor_14','sensor_15',
                'sensor_16','sensor_17','sensor_18','sensor_19','sensor_20',
                'sensor_21']

def load_txt_files(folder_path, keyword):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.endswith('.txt') and keyword.lower() in f.lower()]
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
        if temp_df.shape[1] != len(column_names):
            print(f" Skipping {f} (columns={temp_df.shape[1]})")
            continue
        temp_df.columns = column_names
        
        for col in column_names[1:]:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        df_list.append(temp_df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


def load_rul_files(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.startswith('RUL') and f.endswith('.txt')]
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f, header=None)
        temp_df.columns = ['RUL']
        df_list.append(temp_df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

train_df = load_txt_files(cmap_folder, 'train')
test_df  = load_txt_files(cmap_folder, 'test')
rul_df   = load_rul_files(cmap_folder)


print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("RUL shape:", rul_df.shape)
print("\nTrain preview:\n", train_df.head())
print("\nTest preview:\n", test_df.head())
print("\nRUL preview:\n", rul_df.head())


