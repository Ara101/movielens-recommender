"""
fetch_movielens.py - Download and extract MovieLens 100k dataset.
Saves all data files (u.data, u.item, u.user, u.genre, etc.) to data/ml-100k/.

Run from anywhere:
    python src/fetch_movielens.py
"""

import urllib.request
import zipfile
import os

# Resolve paths relative to the repo root (works regardless of cwd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SCRIPT_DIR, "..")

DATA_DIR = os.path.join(_ROOT_DIR, "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
# The zip contains a top-level ml-100k/ folder, and we extract into data/ml-100k/,
# so files end up at data/ml-100k/ml-100k/u.data — matching train.py's DATA_PATH.
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-100k")
RATINGS_FILE = os.path.join(EXTRACT_DIR, "ml-100k", "u.data")


def download_movielens():
    """Download the MovieLens 100k dataset if not already present."""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, ZIP_PATH)
        print(f"Download complete → {ZIP_PATH}")
    else:
        print(f"Zip already exists at {ZIP_PATH}, skipping download.")


def extract_data():
    """Extract all MovieLens files from the zip into data/ml-100k/."""
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Zip file not found at {ZIP_PATH}. Run download_movielens() first."
        )

    if os.path.exists(RATINGS_FILE):
        print(f"Data already extracted at {EXTRACT_DIR}")
        return

    print("Extracting dataset ...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print(f"Extraction complete → {EXTRACT_DIR}")


def verify():
    """Check that the key data files exist."""
    required = ["u.data", "u.item", "u.user", "u.genre"]
    data_subdir = os.path.join(EXTRACT_DIR, "ml-100k")
    missing = [f for f in required if not os.path.exists(os.path.join(data_subdir, f))]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_subdir}: {missing}")
    print(f"Verified: all required files present in {data_subdir}")


if __name__ == "__main__":
    download_movielens()
    extract_data()
    verify()
    print(f"\nData ready at: {os.path.abspath(EXTRACT_DIR)}")