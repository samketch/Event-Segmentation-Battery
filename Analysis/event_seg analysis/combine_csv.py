import os
import pandas as pd

# Folder containing all corrected event segmentation CSVs
EVENT_SEG_FOLDER = r"C:\Users\Smallwood Lab\friends-event-segmentation\Tasks\event_seg\Processed_data"

# Output path for the master file
OUTPUT_FILE = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\master_data.csv"

def merge_eventseg(folder: str, output_file: str) -> None:
    all_dfs = []
    for fname in os.listdir(folder):
        if fname.endswith("_boundaries.csv"):
            fpath = os.path.join(folder, fname)
            print(f"Reading {fpath}")
            df = pd.read_csv(fpath)
            all_dfs.append(df)

    if not all_dfs:
        print("No CSV files found in folder.")
        return

    new_data = pd.concat(all_dfs, ignore_index=True)

    # Round BoundaryTime(s) to 3 decimal places if present
    if "BoundaryTime(s)" in new_data.columns:
        new_data["BoundaryTime(s)"] = new_data["BoundaryTime(s)"].round(3)

    # If master_data already exists, append only new rows
    if os.path.exists(output_file):
        print(f"Existing master file found at {output_file}, appending new rows...")
        master = pd.read_csv(output_file)
        combined = pd.concat([master, new_data], ignore_index=True).drop_duplicates(
            subset=["ParticipantID", "VideoName", "BoundaryTime(s)"]
        )
    else:
        print("No master file found, creating new one...")
        combined = new_data

    combined = combined.sort_values(by=["ParticipantID", "VideoName", "BoundaryTime(s)"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    combined.to_csv(output_file, index=False)
    print(f"\nMaster data updated at: {output_file}")

if __name__ == "__main__":
    merge_eventseg(EVENT_SEG_FOLDER, OUTPUT_FILE)
