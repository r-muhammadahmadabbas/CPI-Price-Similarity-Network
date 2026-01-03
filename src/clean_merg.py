import pandas as pd
import glob
import os

# ----------------------------
# PARAMETERS
# ----------------------------
base_dir = r"C:\Semester 3\Discrete strucutres\CPI_PROJ\data"
years = [2023, 2024, 2025]

# Standardized city names
cities = [
    "Islamabad", "Rawalpindi", "Gujranwala", "Sialkot", "Lahore", "Faisalabad",
    "Sargodha", "Multan", "Bahawalpur", "Karachi", "Hyderabad", "Sukkur",
    "Larkana", "Peshawar", "Bannu", "Quetta", "Khuzdar"
]

# ----------------------------
# CLEAN COLUMN NAMES
# ----------------------------
def normalize_columns(cols):
    # Remove newlines, hyphens, extra spaces, lowercase for mapping
    return [str(c).replace("\n","").replace("-","").strip() for c in cols]

# ----------------------------
# FUNCTION: CLEAN MONTHLY FILE
# ----------------------------
def clean_monthly_file(file_path):
    # Try reading first 5 rows to detect correct header
    df = None
    for header_row in range(5):
        temp_df = pd.read_excel(file_path, header=header_row)
        temp_cols = normalize_columns(temp_df.columns)
        if "Description" in temp_cols and "Unit" in temp_cols:
            df = temp_df
            df.columns = temp_cols
            break

    if df is None:
        print(f"Error: Could not find proper header in {file_path}")
        return None

    # Map messy city names to standard city names
    col_mapping = {}
    for col in df.columns:
        col_clean = col.replace(" ","").lower()
        for city in cities:
            if city.replace(" ","").lower() in col_clean:
                col_mapping[col] = city
    df = df.rename(columns=col_mapping)

    # Keep only Description, Unit, and standardized cities
    keep_columns = ["Description", "Unit"] + cities
    missing_cols = [c for c in keep_columns if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in {file_path}: {missing_cols}")
    df = df[[c for c in keep_columns if c in df.columns]]

    return df

# ----------------------------
# FUNCTION: MERGE 12 MONTHS OF A YEAR
# ----------------------------
def merge_yearly(year):
    folder = os.path.join(base_dir, str(year))
    all_files = sorted(glob.glob(os.path.join(folder, f"{year}_*.xlsx")))

    monthly_dfs = []
    for file in all_files:
        df = clean_monthly_file(file)
        if df is not None and not df.empty:
            month = os.path.basename(file).split("_")[1].split(".")[0]
            df["Month"] = month
            monthly_dfs.append(df)

    if monthly_dfs:
        yearly_df = pd.concat(monthly_dfs, ignore_index=True)
        output_file = os.path.join(base_dir, f"{year}_cleaned_yearly.xlsx")
        yearly_df.to_excel(output_file, index=False)
        print(f"Yearly file saved: {output_file}")
    else:
        print(f"No valid data to merge for year {year}")

# ----------------------------
# MAIN LOOP
# ----------------------------
for y in years:
    merge_yearly(y)
