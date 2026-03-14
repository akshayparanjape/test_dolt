import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import structured.utils as utils

'''
Creating Homogenizer State = [0, 1, 2] 
'''
df = utils.load_data("amdp_clean_yenju.csv")

df = df[df["Sample Time"].notna()].copy()

columns = ['Date', 'Shift', 'Recipe', 'Batch No.', 'Batch Start Time',
       'Batch End Time', 'Operator Name', 'Equipment(MM/S1/S2/S3)',
       'Weight', 'Temperature', 'RM name', 'Dose Kg', 'Water Kg', 'Dose Time',
       'Sample Time', 'Sample Temperature (C)',
       'MIXER_SCRAPPER.OUTPUT_CURRENT', 'MIXER_SCRAPPER.ACT_RPM',
       'Mixer_Weight.SCALED_OUTPUT', 'MIXER_AGITATOR.OUTPUT_CURRENT',
       'MIXER_AGITATOR.ACT_RPM', 'MIXER_TRANSFER_PUMP_SRU_4.OUTPUT_CURRENT',
       'MIXER_TRANSFER_PUMP_SRU_4.ACT_RPM',
       'MIXER_SILVERSON_PUMP.OUTPUT_CURRENT', 'MIXER_SILVERSON_PUMP.ACT_RPM',
       'BD After RM', 'pH After RM', 'Viscosity After RM',
       'Homogenizer Start Time', 'Homogenizer Run Time (min)']
    
map_shifts = {"A": 0, "B": 1}
df['Shift'] = df['Shift'].map(map_shifts)

map_recipe = {"LBP Lavander Shampoo": 0, "LBP Onion Shampoo": 1}
df['Recipe'] = df['Recipe'].map(map_recipe)

map_equipment = {"MM": 0, "S1": 1, "S2": 2, "S3": 3}
equipment_col = 'Equipment(MM/S1/S2/S3)'
df[equipment_col] = df[equipment_col].map(map_equipment)

# Build operator name mapping from unique values in the file
unique_operators = sorted(df['Operator Name'].dropna().unique())
operator_name = {name: i for i, name in enumerate(unique_operators)}
print(operator_name)
df['Operator Name'] = df['Operator Name'].map(operator_name)

# Build RM name mapping from unique values in the file
unique_rm_names = sorted(df['RM name'].dropna().unique())
rm_name = {name: i for i, name in enumerate(unique_rm_names)}
print(rm_name)
df['RM name'] = df['RM name'].map(rm_name)


target_columns = ['BD After RM', 'Viscosity After RM', 'pH After RM']

feature_cols = [
        'Shift', 'Recipe', 'Operator Name',
        "Weight", 'RM name',
        "Temperature",
        "Dose Kg",
        "Water Kg",
        "Sample Temperature (C)",
        "MIXER_SCRAPPER.OUTPUT_CURRENT",
        "MIXER_SCRAPPER.ACT_RPM",
        "Sample Lapse Time",
    ]


'''
time duration
'''
for col in ["Batch Start Time", "Dose Time", "Sample Time"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

def time_to_minutes(t):
    if pd.isna(t):
        return np.nan
    return t.hour * 60 + t.minute + t.second / 60

df["Batch Start Time"] = pd.to_datetime(df["Batch Start Time"])
df["Dose Time"] = pd.to_datetime(df["Dose Time"])
df["Sample Time"] = pd.to_datetime(df["Sample Time"])

df["Dose Lapse Time"] = df["Dose Time"].apply(time_to_minutes) - df["Batch Start Time"].apply(time_to_minutes)
df["Sample Lapse Time"] = df["Sample Time"].apply(time_to_minutes) - df["Batch Start Time"].apply(time_to_minutes)

df["Homogenizer Start Time"] = pd.to_datetime(df["Homogenizer Start Time"])
df["Homogenizer Stop Time"] = (df["Homogenizer Start Time"] + pd.to_timedelta(df["Homogenizer Run Time (min)"], unit="m"))

df["Homogenizer State"] = np.select(
    [
        df["Sample Time"] < df["Homogenizer Start Time"],
        df["Sample Time"].between(df["Homogenizer Start Time"], df["Homogenizer Stop Time"])
    ],
    [0, 1],
    default=2
)

def homogenizer_duration(row):
    if row["Sample Time"] < row["Homogenizer Start Time"]:
        return 0.0
    elapsed = (row["Sample Time"] - row["Homogenizer Start Time"]).total_seconds() / 60
    run_time = row["Homogenizer Run Time (min)"]
    if pd.isna(elapsed) or pd.isna(run_time):
        return np.nan
    return min(run_time, elapsed)

df["Homogenizer Duration"] = df.apply(homogenizer_duration, axis=1)


for col in (feature_cols + target_columns + ["Batch No."]):
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Drop only those columns which actually exist in df

feature_cols.append("Homogenizer State")
feature_cols.append("Homogenizer Duration")

encodings = {
    "Shift": map_shifts,
    "Recipe": map_recipe,
    "Equipment(MM/S1/S2/S3)": map_equipment,
    "Operator Name": operator_name,
    "RM name": rm_name
}


import json

with open("categorical_mappings.json", "w") as f:
    json.dump(encodings, f, indent=4)


# X and y slices

s = df["BD After RM"]

df["BD After RM"] = s.where(
    ~s.isna(), 
    (s.shift(1) + s.shift(-1)) / 2
)

X = df[df["Equipment(MM/S1/S2/S3)"] == 0][feature_cols + ["Batch No."]].copy()
y = df[df["Equipment(MM/S1/S2/S3)"] == 0][target_columns].values.copy()


export_df = X.copy()
export_df[target_columns] = y.copy()

print(export_df.columns.tolist())

export_df.to_excel(
    "model_training_table.xlsx",
    index=False
)

# raise SystemExit

from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import pandas as pd

df = pd.read_excel("model_training_table.xlsx")
df.to_csv("model_training_table.csv", index=False)

df = utils.load_data("model_training_table.csv")
# print(df.columns.tolist())hom


'''
for BD training
'''
# X and y slices

print("-----------------BD training0------------------------------")

X = df[feature_cols].copy()
y = df[target_columns[0]].copy()
groups = df["Batch No."].copy()

# raise SystemExit
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

gkf = GroupKFold(n_splits=7)

BD_mae_list = []
BD_r2_list = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train model for a single target
    model_BD = DecisionTreeRegressor(
        random_state=42,
        max_depth=7,        # you can tune this
        min_samples_split=6,   # and this
        min_samples_leaf=5     # and this
    )

    model_BD.fit(X_train, y_train)

    # Predict
    y_pred = model_BD.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    BD_mae_list.append(mae)
    BD_r2_list.append(r2)

    test_batch = df.iloc[test_idx]["Batch No."].unique()
    print(f"Fold {fold+1} — Test Batch {test_batch}")
    print(f"  MAE = {mae:.4f}")
    print(f"  R²  = {r2:.4f}")
    print("-" * 40)

'''
for viscosity training
'''
# X and y slices



print("-----------------Viscosity training0------------------------------")
X = df[feature_cols].copy()
y = df[target_columns[1]].copy()
groups = df["Batch No."].copy()

# raise SystemExit
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

gkf = GroupKFold(n_splits=7)

viscosity_mae_list = []
viscosity_r2_list = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train model for a single target
    model_viscosity = DecisionTreeRegressor(
        random_state=42,
        max_depth=7,        # you can tune this
        min_samples_split=6,   # and this6
        min_samples_leaf=5     # and this5
    )

    model_viscosity.fit(X_train, y_train)

    # Predict
    y_pred = model_viscosity.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    viscosity_mae_list.append(mae)
    viscosity_r2_list.append(r2)

    test_batch = df.iloc[test_idx]["Batch No."].unique()
    print(f"Fold {fold+1} — Test Batch {test_batch}")
    print(f"  MAE = {mae:.4f}")
    print(f"  R²  = {r2:.4f}")
    print("-" * 40)



'''
for pH training
'''
# X and y slices

print("-----------------pH training0------------------------------")
X = df[feature_cols].copy()
y = df[target_columns[2]].copy()
groups = df["Batch No."].copy()

# raise SystemExit
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

gkf = GroupKFold(n_splits=7)

pH_mae_list = []
pH_r2_list = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train model for a single target
    model_pH = DecisionTreeRegressor(
        random_state=42,
        max_depth=7,        # you can tune this
        min_samples_split=6,   # and this
        min_samples_leaf=5     # and this
    )

    model_pH.fit(X_train, y_train)

    # Predict
    y_pred = model_pH.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    pH_mae_list.append(mae)
    pH_r2_list.append(r2)

    test_batch = df.iloc[test_idx]["Batch No."].unique()
    print(f"Fold {fold+1} — Test Batch {test_batch}")
    print(f"  MAE = {mae:.4f}")
    print(f"  R²  = {r2:.4f}")
    print("-" * 40)

print("=== Final Leave-One-Batch-Out Results ===")
print(f"BD MAE avg: {np.mean(BD_mae_list):.4f}")
print(f" BD R²  avg: {np.mean(BD_r2_list):.4f}")


print("=== Final Leave-One-Batch-Out Results ===")
print(f"viscosity MAE avg: {np.mean(viscosity_mae_list):.4f}")
print(f" viscosity R²  avg: {np.mean(viscosity_r2_list):.4f}")

print("=== Final Leave-One-Batch-Out Results ===")
print(f"pH MAE avg: {np.mean(pH_mae_list):.4f}")
print(f" pH R²  avg: {np.mean(pH_r2_list):.4f}")

import shap

def shap_for_batch(df, model, feature_cols, title, plot_type):
    X = df[feature_cols]

    # Use background data for initiating the SHAP explainer
    # Select a random sample of the data as background data for the explainer
    background_data = X.sample(100, random_state=42) if len(X) > 100 else X

    # Build SHAP explainer using background data
    explainer = shap.TreeExplainer(model, data=background_data)
    shap_values = explainer.shap_values(X)

    # High resolution plot: Set figure DPI high before plotting
    import matplotlib.pyplot as plt
    plt.figure(dpi=200)  # Set high DPI for high resolution

    shap.summary_plot(
        shap_values,
        X,
        plot_type=plot_type,
        show=False  # delay showing so we can add title
    )

    plt.title(f"Feature Importance — {title}")
    plt.tight_layout()
    plt.savefig(f"decision_tree_shap_{title}_{plot_type}.png", dpi=200)
    plt.show()


shap_for_batch(X, model=model_BD, feature_cols=feature_cols, title="model_BD", plot_type="dot")
shap_for_batch(X, model=model_viscosity, feature_cols=feature_cols, title="model_viscosity", plot_type="dot")
shap_for_batch(X, model=model_pH, feature_cols=feature_cols, title="model_pH", plot_type="dot")

shap_for_batch(X, model=model_BD, feature_cols=feature_cols, title="model_BD", plot_type="bar")
shap_for_batch(X, model=model_viscosity, feature_cols=feature_cols, title="model_viscosity", plot_type="bar")
shap_for_batch(X, model=model_pH, feature_cols=feature_cols, title="model_pH", plot_type="bar")


def plot_tree_model_batch(
    df, model, feature_cols, x_axis_col, y_axis_col, output_var_idx,
    ):

    g = df.copy().sort_values(x_axis_col)
    # Extract inputs
    X_batch = g[feature_cols].copy()
    x = g[x_axis_col].values
    y_true = g[y_axis_col].copy().values

    # Predict
    y_pred = model.predict(X_batch)
    y_pred = y_pred[:, output_var_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_true, color="blue", s=60, label=f"Actual {y_axis_col}")
    plt.plot(x, y_pred, color="red", linewidth=2, label=f"Predicted {y_axis_col}")

    plt.xlabel(x_axis_col)
    plt.ylabel(f"{y_axis_col}")
    plt.title(f"Actual vs Predicted {y_axis_col} (Regressor)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# target_columns = ['BD After RM', 'Viscosity After RM', 'pH After RM']

# plot_tree_model_batch(df, model=model_viscosity, feature_cols=feature_cols, x_axis_col='Sample Lapse Time', y_axis_col='BD After RM', output_var_idx=0)
# plot_tree_model_batch(df, model=model_BD, feature_cols=feature_cols, x_axis_col='Sample Lapse Time', y_axis_col='Viscosity After RM', output_var_idx=1)
# plot_tree_model_batch(df, model=model_pH, feature_cols=feature_cols, x_axis_col='Sample Lapse Time', y_axis_col='pH After RM', output_var_idx=2)



