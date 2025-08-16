import pandas as pd
import os
import joblib

# Allowed moods for the emotion detector
ALLOWED = {"happy","sad","neutral","angry","fear","surprise","disgust"}

def main():
    # Check if songs.csv exist
    if not os.path.exists("songs.csv"):
        raise FileNotFoundError("songs.csv not found. Please create it first.")

    # Load CSV
    df = pd.read_csv("songs.csv")

    # Check required columns
    for col in ["title","artist","url","mood"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Keep only allowed moods
    bad = ~df["mood"].isin(ALLOWED)
    if bad.any():
        print("[WARN] Some rows have unsupported moods. They will be dropped:")
        print(df[bad][["title","mood"]])
        df = df[~bad]

    if df.empty:
        raise ValueError("No valid rows to index. Check moods and CSV content.")

    # Build mood → songs index
    index = {mood: df[df.mood==mood][["title","artist","url"]].to_dict("records")
             for mood in sorted(df["mood"].unique())}

    # Save index to models/song_index.pkl
    os.makedirs("models", exist_ok=True)
    joblib.dump(index, "models/song_index.pkl")
    print(f"[OK] Indexed {len(df)} songs across {len(index)} moods → models/song_index.pkl")

if __name__ == "__main__":
    main()
