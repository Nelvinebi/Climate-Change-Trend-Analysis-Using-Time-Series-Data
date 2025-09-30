"""
Climate-Change-Trend-Analysis-Using-Time-Series-Data
-----------------------------------------------------
This project demonstrates a time-series analysis of synthetic climate variables
(temperature anomaly, CO2 concentration, and sea level rise).
It applies decomposition, visualization, and forecasting models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Generate synthetic time-series data
# -----------------------------
np.random.seed(42)
n_months = 240  # 20 years of monthly data (>100 points)
dates = pd.date_range(start="2000-01-01", periods=n_months, freq="M")

# Synthetic climate variables
# Global temperature anomaly (°C) - rising trend + seasonal cycle + noise
temp_trend = np.linspace(0, 1.2, n_months)
temp_season = 0.2 * np.sin(2 * np.pi * np.arange(n_months) / 12)
temp_noise = np.random.normal(0, 0.05, n_months)
temperature_anomaly = temp_trend + temp_season + temp_noise

# CO2 concentration (ppm) - rising trend with seasonality
co2_trend = 370 + np.linspace(0, 50, n_months)
co2_season = 3 * np.sin(2 * np.pi * np.arange(n_months) / 12)
co2_noise = np.random.normal(0, 1, n_months)
co2_concentration = co2_trend + co2_season + co2_noise

# Sea level rise (cm) - quadratic trend + noise
sea_level_trend = 0.01 * (np.arange(n_months) ** 1.1)
sea_level_noise = np.random.normal(0, 0.5, n_months)
sea_level_rise = sea_level_trend + sea_level_noise

# Build DataFrame
df = pd.DataFrame({
    "date": dates,
    "temperature_anomaly": temperature_anomaly,
    "co2_concentration": co2_concentration,
    "sea_level_rise": sea_level_rise
})
df.set_index("date", inplace=True)

# Save dataset
out_csv = Path("climate_change_time_series.csv")
df.to_csv(out_csv)
print(f"Saved synthetic dataset to {out_csv.resolve()}")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["temperature_anomaly"], label="Temperature Anomaly (°C)")
plt.plot(df.index, df["co2_concentration"]/100, label="CO₂ Concentration (/100 ppm)")
plt.plot(df.index, df["sea_level_rise"]/10, label="Sea Level Rise (/10 cm)")
plt.title("Synthetic Climate Change Indicators (2000–2019)")
plt.xlabel("Year")
plt.ylabel("Scaled values")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Seasonal decomposition (temperature anomaly)
# -----------------------------
decomp = seasonal_decompose(df["temperature_anomaly"], model="additive", period=12)
decomp.plot()
plt.suptitle("Seasonal Decomposition of Temperature Anomaly", y=1.02)
plt.show()

# -----------------------------
# Forecasting with ARIMA (temperature anomaly)
# -----------------------------
train_size = int(len(df) * 0.8)
train, test = df["temperature_anomaly"][:train_size], df["temperature_anomaly"][train_size:]

model = ARIMA(train, order=(2, 1, 2))
fit = model.fit()
forecast = fit.forecast(steps=len(test))

# Evaluation
mae = mean_absolute_error(test, forecast)
rmse = mean_squared_error(test, forecast, squared=False)
print(f"Forecast MAE: {mae:.3f}")
print(f"Forecast RMSE: {rmse:.3f}")

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="orange")
plt.plot(test.index, forecast, label="Forecast", color="red", linestyle="--")
plt.title("ARIMA Forecast of Temperature Anomaly")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend()
plt.tight_layout()
plt.show()
