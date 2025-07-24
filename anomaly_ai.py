import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import zscore


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


telemetry_file = "mock-telemetry-data.json"
simulation_file = "mock-drone-data.json"
anomaly_file = "mock-anomaly-data.json"

telemetry_json = load_json(telemetry_file)
simulation_json = load_json(simulation_file)
anomaly_json = load_json(anomaly_file)


records = []
for d in telemetry_json["data"]:
    records.append({
        "timestamp": pd.to_datetime(d["timestamp"]),
        "speed": d["telemetry"]["speed"],
        "altitude": d["telemetry"]["altitude"],
        "heading": d["telemetry"]["heading"]
    })

df = pd.DataFrame(records)
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)


df["speed_diff"] = df["speed"].diff().fillna(0).abs()
df["altitude_diff"] = df["altitude"].diff().fillna(0).abs()
df["heading_diff"] = df["heading"].diff().fillna(0).abs()


SPEED_THRESHOLD = 10   # m/s
ALTITUDE_THRESHOLD = 5 # m
HEADING_THRESHOLD = 10 # derece

df["speed_anomaly"] = df["speed_diff"] > SPEED_THRESHOLD
df["altitude_anomaly"] = df["altitude_diff"] > ALTITUDE_THRESHOLD
df["heading_anomaly"] = df["heading_diff"] > HEADING_THRESHOLD


print("\n📌 Drone Anomali Raporu\n")
for _, row in df.iterrows():
    ts = row["timestamp"]
    if row["speed_anomaly"]:
        print(f"[{ts}] ⚠ Ani hız değişimi: Δ{row['speed_diff']:.2f} m/s")
    if row["altitude_anomaly"]:
        print(f"[{ts}] ⚠ Ani irtifa değişimi: Δ{row['altitude_diff']:.2f} m")
    if row["heading_anomaly"]:
        print(f"[{ts}] ⚠ Ani rota değişimi: Δ{row['heading_diff']:.2f}°")


print("\n📌 Parça / Sensör Durumu\n")
motors = simulation_json["currentDroneState"]["telemetry"].get("motorRpm", [])
if motors:
    mean_rpm = sum(motors) / len(motors)
    for i, rpm in enumerate(motors):
        if abs(rpm - mean_rpm) > 100:
            print(f"⚠ Motor {i+1} RPM anomalisi: {rpm} RPM (ortalama {mean_rpm:.1f})")
else:
    print("⚠ Motor RPM verisi bulunamadı.")


print("\n📌 Kayıtlı Anomaliler (Loglanan)\n")
for a in anomaly_json.get("anomalies", []):
    print(f"[{a['timestamp']}] {a['type']} ({a['severity']}): {a['description']}")


# =====================
# TensorFlow ile Derin Öğrenme Tabanlı Anomali Tespiti
# =====================
data = df[["speed", "altitude", "heading"]].values

# Ölçekleme
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Zaman serisi için pencereleme fonksiyonu
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

WINDOW_SIZE = 10
sequences = create_sequences(data_scaled, WINDOW_SIZE)

# LSTM Autoencoder Modeli
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, activation='relu', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(16, activation='relu', return_sequences=False),
        tf.keras.layers.RepeatVector(input_shape[0]),
        tf.keras.layers.LSTM(16, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

input_shape = (sequences.shape[1], sequences.shape[2])
model = build_model(input_shape)

# Modeli eğit (örnek: 10 epoch, gerçek uygulamada daha fazla olabilir)
history = model.fit(sequences, sequences, epochs=10, batch_size=32, verbose=0)

# Rekonstrüksiyon hatası ile anomali skoru hesapla
reconstructions = model.predict(sequences)
mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1,2))

# Skorları yüzdeye çevir (min-max normalization ile 0-100 arası)
min_mse = np.min(mse)
max_mse = np.max(mse)
if max_mse > min_mse:
    mse_percent = 100 * (mse - min_mse) / (max_mse - min_mse)
else:
    mse_percent = np.zeros_like(mse)

# Anomali threshold'u (örnek: ortalama + 2*std)
threshold = np.mean(mse) + 2*np.std(mse)
anomaly_flags = mse > threshold

# Sonuçları DataFrame'e ekle (pencere sonuna işaret koyar)
df["deep_anomaly"] = False
df["deep_anomaly_score"] = 0.0
for i, (flag, percent) in enumerate(zip(anomaly_flags, mse_percent)):
    idx = i+WINDOW_SIZE-1
    if idx < len(df):
        df.loc[idx, "deep_anomaly_score"] = percent
        if flag:
            df.loc[idx, "deep_anomaly"] = True

print("\n📌 TensorFlow Derin Öğrenme ile Tespit Edilen Anomaliler\n")
print(f"Anomali threshold'u: {threshold:.6f}")
print(f"Tespit edilen anomali skorları (ilk 20): {mse[:20]}")
print(f"Tespit edilen anomali yüzde skorları (ilk 20): {mse_percent[:20]}")
print(f"Toplam tespit edilen anomali sayısı: {df['deep_anomaly'].sum()}")
anomaly_percent = 100 * df["deep_anomaly"].sum() / len(df)
print(f"Anomali oranı (yüzde): {anomaly_percent:.2f}%")
if df["deep_anomaly"].sum() == 0:
    print("⚠ Derin öğrenme modeli mevcut veride anomali tespit etmedi.")
for i, row in df[df["deep_anomaly"]].iterrows():
    print(f"[{row['timestamp']}] ⚠ Derin öğrenme ile anomali tespit edildi! Skor: {row['deep_anomaly_score']:.2f}%")

# LINEER REGRESYON TABANLI ANOMALİ
for col in ["speed", "altitude", "heading"]:
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[col].values
    model_lr = LinearRegression().fit(X, y)
    y_pred = model_lr.predict(X)
    df[f"{col}_linear_resid"] = np.abs(y - y_pred)
    resid_norm = (df[f"{col}_linear_resid"] - df[f"{col}_linear_resid"].min()) / (df[f"{col}_linear_resid"].max() - df[f"{col}_linear_resid"].min() + 1e-8)
    df[f"{col}_linear_score"] = 100 * resid_norm

# LOGARİTMİK DEĞİŞİM TABANLI ANOMALİ
for col in ["speed", "altitude", "heading"]:
    log_val = np.log(df[col].replace(0, np.nan).fillna(1))
    log_diff = log_val.diff().abs().fillna(0)
    log_norm = (log_diff - log_diff.min()) / (log_diff.max() - log_diff.min() + 1e-8)
    df[f"{col}_log_score"] = 100 * log_norm

# Z-SCORE TABANLI ANOMALİ
for col in ["speed", "altitude", "heading"]:
    z = np.abs(zscore(df[col].values))
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    df[f"{col}_zscore"] = 100 * z_norm

# GAUSSIAN MIXTURE MODEL TABANLI ANOMALİ
for col in ["speed", "altitude", "heading"]:
    vals = df[col].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(vals)
    log_probs = -gmm.score_samples(vals)
    gmm_norm = (log_probs - log_probs.min()) / (log_probs.max() - log_probs.min() + 1e-8)
    df[f"{col}_gmm_score"] = 100 * gmm_norm

# ENSEMBLE (ORTALAMA) ANOMALİ SKORU
ensemble_scores = []
for i in range(len(df)):
    scores = []
    # Lineer, log, zscore, gmm, deep anomaly score
    for col in ["speed", "altitude", "heading"]:
        scores.append(df.loc[i, f"{col}_linear_score"])
        scores.append(df.loc[i, f"{col}_log_score"])
        scores.append(df.loc[i, f"{col}_zscore"])
        scores.append(df.loc[i, f"{col}_gmm_score"])
    scores.append(df.loc[i, "deep_anomaly_score"])
    ensemble_scores.append(np.mean(scores))
df["ensemble_anomaly_score"] = ensemble_scores

# Genel anomali oranı: tüm ensemble_anomaly_score'ların ortalaması
ensemble_percent = np.mean(df["ensemble_anomaly_score"])
print("\n📌 ENSEMBLE (ORTALAMA) ANOMALİ SKORU VE YÜZDESİ\n")
print(df[["timestamp", "ensemble_anomaly_score"]].tail(20))
print(f"Genel anomali oranı (ortalama yüzde): {ensemble_percent:.2f}%")

# Her satır için yorum
for i, row in df.iterrows():
    if row["ensemble_anomaly_score"] > ensemble_percent:
        print(f"[{row['timestamp']}] ⚠ ENSEMBLE: Ortalama üstü anomali! Skor: {row['ensemble_anomaly_score']:.2f}%")
    # else: # İsterseniz ortalama altı için de bilgi verebilirsiniz
    #     print(f"[{row['timestamp']}] Normal. Skor: {row['ensemble_anomaly_score']:.2f}%")

plt.figure(figsize=(12,6))
plt.plot(df["timestamp"], df["speed"], label="Speed (m/s)", color="blue")
plt.plot(df["timestamp"], df["altitude"], label="Altitude (m)", color="green")
plt.plot(df["timestamp"], df["heading"], label="Heading (°)", color="orange")

# Klasik anomali çizgileri
anomaly_idx = df[(df["speed_anomaly"]) | (df["altitude_anomaly"]) | (df["heading_anomaly"])].index
for i in anomaly_idx:
    plt.axvline(df.loc[i, "timestamp"], color="red", linestyle="--", alpha=0.5)

# ENSEMBLE kritik anomali noktaları (skor > 50)
ensemble_critical_idx = df[df["ensemble_anomaly_score"] > 50].index
for i in ensemble_critical_idx:
    plt.axvline(df.loc[i, "timestamp"], color="red", linestyle="-", alpha=0.9, linewidth=2)

plt.legend()
plt.title("Drone Telemetry & Anomalies")
plt.xlabel("Time")
plt.ylabel("Values")
plt.grid(True)
plt.tight_layout()
plt.show()