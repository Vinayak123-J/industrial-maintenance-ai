from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
TARGET = DATA / "ai4i2020.csv"
URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

if TARGET.exists():
    print(f"Dataset already exists: {TARGET}")
else:
    print("Downloading AI4I 2020 dataset from the UCI Machine Learning Repository...")
    with urlopen(URL, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))
    csv_name = next((n for n in archive.namelist() if n.lower().endswith("ai4i2020.csv")), None)
    if not csv_name:
        raise RuntimeError("Could not find ai4i2020.csv inside the downloaded archive.")
    TARGET.write_bytes(archive.read(csv_name))
    print(f"Saved dataset: {TARGET}")
