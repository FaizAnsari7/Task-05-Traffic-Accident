
"""
Task-05: Traffic Accident Data Analysis
Goal: Analyze traffic accidents to identify patterns related
to weather, time of day, severity, and accident hotspots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1. Load data ----------
csv_path = "us_accidents.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Start_Time"])
else:
    print(f"'{csv_path}' not found — generating a synthetic accidents sample.")

    rng = np.random.default_rng(11)
    n = 1000
    hours = rng.integers(0, 24, n)

    # Bias more accidents to morning/evening rush hours
    rush_boost = np.isin(hours, [7, 8, 9, 16, 17, 18]).astype(int)

    df = pd.DataFrame({
        "Start_Time": (
            pd.to_datetime("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
            + pd.to_timedelta(hours, unit="h")
        ),
        "Hour": hours,
        "Weather_Condition": rng.choice(
            ["Clear", "Rain", "Fog", "Snow", "Cloudy"],
            n, p=[0.45, 0.2, 0.1, 0.1, 0.15]
        ),
        "Severity": rng.choice(
            [1, 2, 3, 4], n, p=[0.2, 0.5, 0.22, 0.08]
        ),
        # Approximate accident hotspot coordinates
        "Start_Lat": rng.normal(34.05, 0.15, n) + rush_boost * 0.02,
        "Start_Lng": rng.normal(-118.25, 0.15, n),
    })

print(df.head())
print("\nShape:", df.shape)

# ---------- 2. Derive time-of-day feature ----------
if "Hour" not in df.columns and "Start_Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Start_Time"]).dt.hour


def time_bucket(h):
    if 5 <= h < 12:
        return "Morning"
    elif 12 <= h < 17:
        return "Afternoon"
    elif 17 <= h < 21:
        return "Evening"
    return "Night"


df["Time_of_Day"] = df["Hour"].apply(time_bucket)

# ---------- 3. Visualizations ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Accidents by hour of day
df["Hour"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0, 0], color="#4C72B0"
)
axes[0, 0].set_title("Accidents by Hour of Day")
axes[0, 0].set_xlabel("Hour")
axes[0, 0].set_ylabel("Number of Accidents")

# Accidents by weather condition
df["Weather_Condition"].value_counts().plot(
    kind="bar", ax=axes[0, 1], color="#DD8452"
)
axes[0, 1].set_title("Accidents by Weather Condition")
axes[0, 1].set_ylabel("Number of Accidents")

# Severity by time of day
pivot = pd.crosstab(df["Time_of_Day"], df["Severity"])

sns.heatmap(
    pivot, annot=True, fmt="d",
    cmap="YlOrRd", ax=axes[1, 0]
)
axes[1, 0].set_title("Severity Counts by Time of Day")

# Accident hotspots by latitude and longitude
scatter = axes[1, 1].scatter(
    df["Start_Lng"],
    df["Start_Lat"],
    c=df["Severity"],
    cmap="Reds",
    alpha=0.5,
    s=10
)

axes[1, 1].set_title("Accident Hotspots (colored by severity)")
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")

plt.colorbar(scatter, ax=axes[1, 1], label="Severity")

plt.tight_layout()

# ---------- 4. Save output ----------
out_path = "task5_traffic_accidents.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved charts to {out_path}")

# ---------- 5. Key insights ----------
print("\nTop weather conditions in accidents:\n",
      df["Weather_Condition"].value_counts())

print("\nAccidents by time of day:\n",
      df["Time_of_Day"].value_counts())

print("\nAverage severity by weather:\n",
      df.groupby("Weather_Condition")["Severity"]
      .mean().sort_values(ascending=False))

plt.show()
