"""Train and evaluate the BWS emulator, then run the 36 scenario predictions"""

# Import libraries and mount Drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

from google.colab import drive
drive.mount('/content/drive')

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lightgbm import LGBMRegressor, early_stopping, log_evaluation


# Project paths and expected files

drive_root_candidates = [
    Path("/content/drive/MyDrive"),
    Path("/content/drive/My Drive"),
]

DRIVE_ROOT = next(
    (path for path in drive_root_candidates if path.exists()),
    None,
)


PROJECT_ROOT = DRIVE_ROOT / "BWS_Emulator_Data"

TRAINING_CSV = PROJECT_ROOT / "Training" / "training_data.csv"

CATEGORY_NAMES = (
    "Baseline",
    "Climate_Sensitivity",
    "Demand_Sensitivity",
)

PREDICTION_INPUT_ROOT = PROJECT_ROOT / "Prediction_input"
PREDICTION_OUTPUT_ROOT = PROJECT_ROOT / "Prediction_output"

EXPECTED_SSPS = (1, 2, 5)

EXPECTED_PERIODS = (
    (2021, 2040),
    (2041, 2060),
    (2061, 2080),
    (2081, 2100),
)

EXPECTED_SCENARIO_FILENAMES = [
    f"SSP{ssp}_{start_year}_{end_year}.csv"
    for ssp in EXPECTED_SSPS
    for start_year, end_year in EXPECTED_PERIODS
]


# Load data

if not TRAINING_CSV.exists():
    raise FileNotFoundError(f"Training file not found: {TRAINING_CSV}")

original_df = pd.read_csv(TRAINING_CSV)

# Check data types
print(original_df.info())


# Basic data checks

print("\nData Preview:")
print(original_df.head(50))

print("\nDataset Information:")
print(original_df.info())

print(f"\nShape: {original_df.shape}")

print("\nMissing Values:")
print(original_df.isnull().sum()[original_df.isnull().sum() > 0])

print("\nStatistical Summary:")
print(original_df.describe())

# Count values greater than or equal to 1
greater_than_one_counts = (original_df >= 1).sum()

print("\nNumber of values >= 1 per column:")
print(greater_than_one_counts)


# Validate training data structure

required_training_columns = {"lon", "lat"}

for month in range(1, 13):
    required_training_columns.update(
        {
            f"bws_{month:02}",
            f"wd_{month:02}",
            f"precipitation_{month:02}",
            f"Temperature{month:02}",
        }
    )

missing_training_columns = required_training_columns - set(original_df.columns)

if missing_training_columns:
    raise ValueError(
        "The training dataset is missing required columns: "
        f"{sorted(missing_training_columns)}"
    )

duplicate_training_locations = int(original_df.duplicated(subset=["lon", "lat"]).sum())

if duplicate_training_locations:
    raise ValueError(
        "The training dataset contains "
        f"{duplicate_training_locations} duplicated lon/lat locations. "
        "The original group definition assumes one row per grid cell."
    )

# Keep coordinates for later spatial analyses, not as direct model inputs
location_metadata = original_df[["lon", "lat"]].copy()
location_metadata["group"] = original_df.index


# Build monthly features and targets

def assign_season(month, lat):
    """Assign a season number from the month and latitude"""
    if lat >= 0:
        if month in [12, 1, 2]:
            return 1
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        elif month in [9, 10, 11]:
            return 4
    else:
        if month in [12, 1, 2]:
            return 3
        elif month in [3, 4, 5]:
            return 4
        elif month in [6, 7, 8]:
            return 1
        elif month in [9, 10, 11]:
            return 2


def season_from_row(row, month):
    return assign_season(month, row["lat"])


features_df = pd.DataFrame()
target_df = pd.DataFrame()

for month in range(1, 13):
    bws_col = f"bws_{month:02}"
    wd_col = f"wd_{month:02}"
    precip_col = f"precipitation_{month:02}"
    temp_col = f"Temperature{month:02}"

    if (
        bws_col in original_df.columns
        and wd_col in original_df.columns
        and precip_col in original_df.columns
        and temp_col in original_df.columns
    ):
        temp_features = original_df[["lon", "lat"]].copy()

        temp_features["month"] = month
        temp_features["wd"] = original_df[wd_col]
        temp_features["precipitation"] = original_df[precip_col]
        temp_features["temperature"] = original_df[temp_col]

        temp_features["season"] = temp_features.apply(
            season_from_row, axis=1, args=(month,)
        )

        # Keep all months from one grid cell in the same group
        temp_features["group"] = original_df.index

        temp_target = original_df[bws_col]

        features_df = pd.concat([features_df, temp_features], ignore_index=True)

        target_df = pd.concat([target_df, temp_target], ignore_index=True)

    else:
        print(f"Missing data for month {month:02}. Skipping this month.")

target_df.columns = ["bws"]

print("\nFeatures DataFrame (sample):")
print(features_df.head(25))

print("\nTarget DataFrame (sample):")
print(target_df.head(25))


# Encode month as cyclical features

features_df["month_sin"] = np.sin(2 * np.pi * features_df["month"] / 12)

features_df["month_cos"] = np.cos(2 * np.pi * features_df["month"] / 12)

features_df.drop(columns=["month"], inplace=True)

print("\nFeatures DataFrame after adding cyclical month features:")
print(features_df.head(25))


# Convert coordinates to 3D Cartesian features

if "lon" in features_df.columns and "lat" in features_df.columns:
    features_df["lat_rad"] = np.radians(features_df["lat"])

    features_df["lon_rad"] = np.radians(features_df["lon"])

    features_df["x"] = (
        np.cos(features_df["lat_rad"])
        * np.cos(features_df["lon_rad"])
    )

    features_df["y"] = (
        np.cos(features_df["lat_rad"])
        * np.sin(features_df["lon_rad"])
    )

    features_df["z"] = np.sin(features_df["lat_rad"])

    features_df.drop(columns=["lon", "lat", "lat_rad", "lon_rad"], inplace=True)

else:
    print("Error: 'lon' and 'lat' columns are missing in features_df.")

# Spatial interactions
features_df["x_y"] = features_df["x"] * features_df["y"]
features_df["x_z"] = features_df["x"] * features_df["z"]
features_df["y_z"] = features_df["y"] * features_df["z"]

# Water-demand interactions
features_df["wd_x"] = features_df["wd"] * features_df["x"]
features_df["wd_y"] = features_df["wd"] * features_df["y"]
features_df["wd_z"] = features_df["wd"] * features_df["z"]
features_df["wd_season"] = features_df["wd"] * features_df["season"]
features_df["wd_month_sin"] = features_df["wd"] * features_df["month_sin"]
features_df["wd_month_cos"] = features_df["wd"] * features_df["month_cos"]

# Precipitation interactions
features_df["precipitation_x"] = features_df["precipitation"] * features_df["x"]
features_df["precipitation_y"] = features_df["precipitation"] * features_df["y"]
features_df["precipitation_z"] = features_df["precipitation"] * features_df["z"]
features_df["precipitation_season"] = (
    features_df["precipitation"] * features_df["season"]
)
features_df["precipitation_month_sin"] = (
    features_df["precipitation"] * features_df["month_sin"]
)
features_df["precipitation_month_cos"] = (
    features_df["precipitation"] * features_df["month_cos"]
)

# Temperature interactions
features_df["temperature_x"] = features_df["temperature"] * features_df["x"]
features_df["temperature_y"] = features_df["temperature"] * features_df["y"]
features_df["temperature_z"] = features_df["temperature"] * features_df["z"]
features_df["temperature_season"] = features_df["temperature"] * features_df["season"]
features_df["temperature_month_sin"] = (
    features_df["temperature"] * features_df["month_sin"]
)
features_df["temperature_month_cos"] = (
    features_df["temperature"] * features_df["month_cos"]
)

print(
    "\nFeatures DataFrame after 3D Cartesian transformation "
    "and adding interactions (sample):"
)
print(features_df.head(30))

print("\nShape of features_df:", features_df.shape)


# Split train, validation, and test data by location

merged_df = features_df.copy()
merged_df["bws"] = target_df["bws"]

print("\nMerged DataFrame shape:", merged_df.shape)

# First split: 80% train and validation, 20% test
gss = GroupShuffleSplit(test_size=0.2, random_state=42)

groups = merged_df["group"]

train_val_idx, test_idx = next(gss.split(merged_df, groups=groups))

train_val_data = merged_df.iloc[train_val_idx].copy()

test_data = merged_df.iloc[test_idx].copy()

print("\nAfter first group-based split:")
print("Train+Validation shape:", train_val_data.shape)
print("Test shape:", test_data.shape)

# Second split: 60% training and 20% validation overall
gss_val = GroupShuffleSplit(test_size=0.25, random_state=42)

groups_train_val = train_val_data["group"]

train_idx, val_idx = next(gss_val.split(train_val_data, groups=groups_train_val))

train_data = train_val_data.iloc[train_idx].copy()

val_data = train_val_data.iloc[val_idx].copy()

print("\nAfter second group-based split:")
print("Training shape:", train_data.shape)
print("Validation shape:", val_data.shape)

# Confirm that grid-cell groups do not overlap
train_groups = set(train_data["group"])
val_groups = set(val_data["group"])
test_groups = set(test_data["group"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Separate features and targets
X_train = train_data.drop(["bws", "group"], axis=1)
y_train = train_data["bws"]

X_val = val_data.drop(["bws", "group"], axis=1)
y_val = val_data["bws"]

X_test = test_data.drop(["bws", "group"], axis=1)
y_test = test_data["bws"]

print("\nFinal dataset shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)


# Train LightGBM

model = LGBMRegressor(
    learning_rate=0.01,
    n_estimators=50000,
    num_leaves=80,
    max_depth=16,
    colsample_bytree=0.8,
    reg_alpha=0.9,
    reg_lambda=0.8,
    min_child_samples=20,
    random_state=42,
)

# Store the evaluation history
evals_result = defaultdict(list)


def record_evaluation_result(env):
    for data_name, eval_name, result, _ in env.evaluation_result_list:
        evals_result[f"{data_name}_{eval_name}"].append(result)


model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],
    eval_metric="rmse",
    callbacks=[
        early_stopping(stopping_rounds=500, verbose=True),
        log_evaluation(period=100),
        record_evaluation_result,
    ],
)

print("\nEvaluation history (first few entries):")
for key, values in evals_result.items():
    print(f"{key}: {values[:5]} ...")


# Evaluate test performance

# Keep test predictions unchanged
if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
else:
    y_test_pred = model.predict(X_test)

y_test = np.array(y_test).ravel()
y_test_pred = np.array(y_test_pred).ravel()

# Report negative test predictions without changing them
negative_test_prediction_count = int(np.sum(y_test_pred < 0))

print(
    "\nNegative test predictions "
    "(reported only; not modified):",
    negative_test_prediction_count,
)

# Calculate the original metrics
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
var_y_test = np.var(y_test) if np.var(y_test) != 0 else np.nan
nmse = mse / var_y_test
denom_nse = np.sum((y_test - np.mean(y_test)) ** 2)

nse = (
    1 - np.sum((y_test - y_test_pred) ** 2) / denom_nse
    if denom_nse != 0
    else np.nan
)

sum_y_test = np.sum(y_test)

pbias = (
    100 * np.sum(y_test - y_test_pred) / sum_y_test
    if sum_y_test != 0
    else np.nan
)

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Error (MAE)",
            "R-squared (R²)",
            "Normalized MSE (NMSE)",
            "Nash-Sutcliffe Efficiency (NSE)",
            "Percent Bias (PBIAS)",
        ],
        "Value": [mse, rmse, mae, r2, nmse, nse, pbias],
    }
)

print("\nModel Performance on Test Set:")
print(metrics_df)


# Validate the 36 scenario input files

def validate_scenario_folders():
    """Check the 12 expected files in each scenario category"""
    expected_csv_names = set(EXPECTED_SCENARIO_FILENAMES)

    for category_name in CATEGORY_NAMES:
        input_dir = PREDICTION_INPUT_ROOT / category_name

        if not input_dir.exists():
            raise FileNotFoundError(f"Prediction input folder not found: {input_dir}")

        found_csv_names = {path.name for path in input_dir.glob("*.csv")}

        missing_files = expected_csv_names - found_csv_names

        if missing_files:
            raise ValueError(
                f"{category_name} is missing files: "
                f"{sorted(missing_files)}"
            )

        output_dir = PREDICTION_OUTPUT_ROOT / category_name

        output_dir.mkdir(parents=True, exist_ok=True)

    print("\nScenario-folder check passed: 12 required files in each category.")


validate_scenario_folders()


# Predict monthly BWS for one scenario file

# Use the training feature order
feature_cols = list(X_train.columns)


def predict_12month_bws(
    model,
    csv_in,
    csv_out,
    feature_cols,
    batch_size=5000,
):
    """Predict 12 monthly BWS values for one scenario file"""
    csv_in = Path(csv_in)
    csv_out = Path(csv_out)

    df_in = pd.read_csv(csv_in)

    print("\nInput:", csv_in)
    print("Input rows:", len(df_in))

    required_prediction_columns = {"lon", "lat"}

    for month in range(1, 13):
        required_prediction_columns.update(
            {
                f"wd_{month:02}",
                f"Temperature_{month:02}",
                f"Precipitation_{month:02}",
            }
        )

    missing_prediction_columns = required_prediction_columns - set(df_in.columns)

    if missing_prediction_columns:
        raise ValueError(
            f"{csv_in.name} is missing required columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    duplicate_prediction_locations = int(df_in.duplicated(subset=["lon", "lat"]).sum())

    if duplicate_prediction_locations:
        raise ValueError(
            f"{csv_in.name} contains "
            f"{duplicate_prediction_locations} duplicated "
            "lon/lat locations."
        )

    features_all = []
    meta_all = []

    for month in range(1, 13):
        wd_col = f"wd_{month:02}"
        temp_col = f"Temperature_{month:02}"
        precip_col = f"Precipitation_{month:02}"

        f = df_in[["lon", "lat"]].copy()

        f["month"] = month
        f["wd"] = df_in[wd_col]
        f["temperature"] = df_in[temp_col]
        f["precipitation"] = df_in[precip_col]

        f["season"] = f.apply(season_from_row, axis=1, args=(month,))

        # Cyclical month features
        f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)

        f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)

        f.drop(columns=["month"], inplace=True)

        # 3D coordinates
        lat_r = np.radians(f["lat"])
        lon_r = np.radians(f["lon"])

        f["x"] = np.cos(lat_r) * np.cos(lon_r)

        f["y"] = np.cos(lat_r) * np.sin(lon_r)

        f["z"] = np.sin(lat_r)

        f.drop(columns=["lon", "lat"], inplace=True)

        # Spatial interactions
        f["x_y"] = f["x"] * f["y"]
        f["x_z"] = f["x"] * f["z"]
        f["y_z"] = f["y"] * f["z"]

        # Water-demand, precipitation, and temperature interactions
        for variable in ["wd", "precipitation", "temperature"]:
            f[f"{variable}_x"] = f[variable] * f["x"]
            f[f"{variable}_y"] = f[variable] * f["y"]
            f[f"{variable}_z"] = f[variable] * f["z"]
            f[f"{variable}_season"] = f[variable] * f["season"]
            f[f"{variable}_month_sin"] = f[variable] * f["month_sin"]
            f[f"{variable}_month_cos"] = f[variable] * f["month_cos"]

        features_all.append(f)

        meta_all.append(df_in[["lon", "lat"]].assign(month=month))

    X_new = pd.concat(features_all, ignore_index=True)

    meta_df = pd.concat(meta_all, ignore_index=True)

    missing_model_features = set(feature_cols) - set(X_new.columns)

    unexpected_model_features = set(X_new.columns) - set(feature_cols)

    if missing_model_features or unexpected_model_features:
        raise ValueError(
            "Prediction-feature mismatch. "
            f"Missing: {sorted(missing_model_features)}; "
            f"Unexpected: {sorted(unexpected_model_features)}"
        )

    # Match the training column order
    X_new = X_new[feature_cols]

    # Predict in batches
    batch = batch_size
    preds = []

    for i in tqdm(range(0, len(X_new), batch), desc=f"Predict {csv_in.stem}"):
        preds.append(model.predict(X_new.iloc[i:i + batch]))

    preds = np.concatenate(preds)

    # Clip only negative future BWS predictions
    negative_prediction_count = int(np.sum(preds < 0))

    preds = np.maximum(preds, 0.0)

    # Convert predictions from long to wide format
    long_df = meta_df.copy()
    long_df["predicted_bws"] = preds

    duplicate_long_rows = int(long_df.duplicated(subset=["lon", "lat", "month"]).sum())

    if duplicate_long_rows:
        raise ValueError(
            f"{duplicate_long_rows} duplicated "
            "lon/lat/month rows were generated."
        )

    wide_values = long_df.pivot(
        index=["lon", "lat"], columns="month", values="predicted_bws"
    ).reset_index()

    wide_values.columns = ["lon", "lat"] + [
        f"bws_{month:02}" for month in range(1, 13)
    ]

    # Keep the input coordinate order
    wide_df = (
        df_in[["lon", "lat"]]
        .merge(
            wide_values,
            on=["lon", "lat"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    expected_output_columns = ["lon", "lat"] + [
        f"bws_{month:02}" for month in range(1, 13)
    ]

    if list(wide_df.columns) != expected_output_columns:
        raise ValueError("Unexpected output-column structure.")

    if len(wide_df) != len(df_in):
        raise ValueError("Output row count does not match the input row count.")

    if not wide_df[["lon", "lat"]].equals(df_in[["lon", "lat"]]):
        raise ValueError(
            "Output coordinate order does not match the input coordinate order."
        )

    csv_out.parent.mkdir(parents=True, exist_ok=True)

    wide_df.to_csv(csv_out, index=False)

    print("Saved:", csv_out)
    print("Output shape:", wide_df.shape)
    print(
        "Negative future predictions replaced with zero:",
        negative_prediction_count,
    )

    return csv_out


# Run all 36 files sequentially

generated_output_files = []

for category_name in CATEGORY_NAMES:
    input_dir = PREDICTION_INPUT_ROOT / category_name

    output_dir = PREDICTION_OUTPUT_ROOT / category_name

    print("\n" + "=" * 72)
    print("Starting category:", category_name)
    print("=" * 72)

    for filename in EXPECTED_SCENARIO_FILENAMES:
        input_file = input_dir / filename

        output_file = output_dir / filename

        generated_output_file = predict_12month_bws(
            model=model,
            csv_in=input_file,
            csv_out=output_file,
            feature_cols=feature_cols,
            batch_size=5000,
        )

        generated_output_files.append(generated_output_file)

    print("Completed category:", category_name)

if len(generated_output_files) != 36:
    raise RuntimeError("The workflow did not generate exactly 36 output files.")

print("\nAll scenario predictions completed:", len(generated_output_files))
