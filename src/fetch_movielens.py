"""
fetch_movielens.py - Download and extract MovieLens 100k dataset.
Saves the main rating file (u.data) to the data/ directory.
"""

import urllib.request
import zipfile
import os
from pathlib import Path

# Constants
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-100k.zip"
EXTRACT_DIR = DATA_DIR / "ml-100k"
RATINGS_FILE = "u.data"  # main file we need
TARGET_FILE = DATA_DIR / "ratings.csv"  # optional: save as CSV for easier use

def download_movielens():
    """Download the MovieLens 100k dataset if not already present."""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download only if not already downloaded
    if not ZIP_PATH.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip file already exists, skipping download.")

def extract_ratings():
    """Extract u.data from the zip and save it as CSV in the data folder."""
    if not ZIP_PATH.exists():
        raise FileNotFoundError("Zip file not found. Run download first.")
    
    # Extract only u.data if not already extracted
    ratings_extracted = EXTRACT_DIR / RATINGS_FILE
    if not ratings_extracts.exists():
        print("Extracting u.data...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extract(f"ml-100k/{RATINGS_FILE}", DATA_DIR)
        print("Extraction complete.")
    else:
        print("u.data already extracted.")
    
    # Optional: Convert to CSV for easier use with pandas
    import csv
    with open(ratings_extracted, 'r') as infile, open(TARGET_FILE, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile)
        writer.writerow(['user_id', 'item_id', 'rating', 'timestamp'])
        writer.writerows(reader)
    print(f"Saved ratings as CSV: {TARGET_FILE}")

if __name__ == "__main__":
    download_movielens()
    extract_ratings()
    print("✅ Data ready! File saved to:", TARGET_FILE)