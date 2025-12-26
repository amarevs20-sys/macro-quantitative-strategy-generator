import os
print("fetch_data.py is located at:")
print(os.path.abspath(__file__))
from fredapi import Fred
import pandas as pd

# 1. Get the directory this file lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the data folder path
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# 3. Create the data folder if it does not exist
os.makedirs(DATA_DIR, exist_ok=True)

# 4. Connect to FRED
fred = Fred(api_key="a91a00f5e6f08693e2f15d0b0fd8682b")

# 5. Define macroeconomic series
series = {
    "CPI": "CPIAUCSL",
    "UNEMPLOYMENT": "UNRATE",
    "FED_FUNDS": "FEDFUNDS",
    "GDP": "GDP"
}

# 6. Download the data
data = {}
for name, code in series.items():
    data[name] = fred.get_series(code)

# 7. Convert to DataFrame
df = pd.DataFrame(data)

# 8. Drop rows with missing values
df = df.dropna()

# 9. Save to CSV
file_path = os.path.join(DATA_DIR, "macro_data.csv")
df.to_csv(file_path)

print("Macro data saved to:", file_path)
fred.get_series("CPIAUCSL")