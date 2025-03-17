import os
import pandas as pd

def convert_txt_to_csv(txt_path, csv_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Extracting familiarity data
    timestamp, left_fam, right_fam = [], [], []
    for line in lines:
        if "Familiarity for left view" in line:
            time = line.split()[0]
            fam_value = float(line.split(':')[-1].strip())
            timestamp.append(time)
            left_fam.append(fam_value)
        elif "Familiarity for right view" in line:
            fam_value = float(line.split(':')[-1].strip())
            right_fam.append(fam_value)
    
    # Convert to dataframe and save as csv
    df = pd.DataFrame({
        "Timestamp": timestamp,
        "Left Eye Familiarity": left_fam,
        "Right Eye Familiarity": right_fam
    })
    
    df.to_csv(csv_path, index=False)

# Directory containing the .txt files
directory = os.path.dirname(os.path.abspath(__file__))

# Convert each .txt file in the directory to .csv
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        txt_path = os.path.join(directory, filename)
        csv_filename = filename.replace(".txt", ".csv")
        csv_path = os.path.join(directory, csv_filename)
        convert_txt_to_csv(txt_path, csv_path)
