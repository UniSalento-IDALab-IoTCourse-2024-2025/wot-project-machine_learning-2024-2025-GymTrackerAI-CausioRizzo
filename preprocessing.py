import pandas as pd
import numpy as np
import os

def calculate_zero_crossing(data):
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    return len(zero_crossings)

# Righe sul quale calcolare i parametri
n_rows = 60

new_df = pd.DataFrame(columns=["mean_x", "mean_y", "mean_z", "var_x", "var_y", "var_z", "peak_to_peak_x", "peak_to_peak_y", "peak_to_peak_z", "zero_crossings_x", "zero_crossings_y", "zero_crossings_z", "Activity"])
j=0

for file in os.listdir('dataset'):
    if file.endswith('.csv'):
        
        df = pd.read_csv(f'dataset/{file}', sep=';')
        
        for i in range(0, df.shape[0] + 1, n_rows):
            
            print(f"\nProcessing {file} from {i} to {i + n_rows - 1}\n")
            
            x_rows = df.loc[i:i + n_rows - 1, "X (mg)"]
            y_rows = df.loc[i:i + n_rows - 1, "Y (mg)"]
            z_rows = df.loc[i:i + n_rows - 1, "Z (mg)"]
            
            if not x_rows.empty and not y_rows.empty and not z_rows.empty:

                mean_x = round(np.mean(x_rows), 2)
                var_x = round(np.var(x_rows), 2)
                peak_to_peak_x = max(x_rows) - min(x_rows)
                zero_crossings_x = calculate_zero_crossing(x_rows)
            
                mean_y = round(np.mean(y_rows), 2)
                var_y = round(np.var(y_rows), 2)
                peak_to_peak_y = max(y_rows) - min(y_rows)
                zero_crossings_y = calculate_zero_crossing(y_rows)
                
                mean_z = round(np.mean(z_rows), 2)
                var_z = round(np.var(z_rows), 2)
                peak_to_peak_z = max(z_rows) - min(z_rows)
                zero_crossings_z = calculate_zero_crossing(z_rows)

                filename_label = file.split('.')[0].lower()

                if filename_label == "plank":
                    activity = "Plank"
                elif filename_label == "jumpingjack":
                    activity = "JumpingJack"
                elif filename_label == "squatjack":
                    activity = "SquatJack"
                else:
                    activity = filename_label.capitalize()

                new_row = pd.DataFrame({
                    "mean_x": mean_x,
                    "mean_y": mean_y,
                    "mean_z": mean_z,
                    "var_x": var_x,
                    "var_y": var_y,
                    "var_z": var_z,
                    "peak_to_peak_x": peak_to_peak_x,
                    "peak_to_peak_y": peak_to_peak_y,
                    "peak_to_peak_z": peak_to_peak_z,
                    "zero_crossings_x": zero_crossings_x,
                    "zero_crossings_y": zero_crossings_y,
                    "zero_crossings_z": zero_crossings_z,
                    "Activity": activity
                }, index=[j])
                j+=1
                
                new_df = pd.concat([new_df, new_row])

new_df.to_csv("dataset_index.csv", sep=';', index=False)

