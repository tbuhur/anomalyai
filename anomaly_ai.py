import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt


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


plt.figure(figsize=(12,6))
plt.plot(df["timestamp"], df["speed"], label="Speed (m/s)", color="blue")
plt.plot(df["timestamp"], df["altitude"], label="Altitude (m)", color="green")
plt.plot(df["timestamp"], df["heading"], label="Heading (°)", color="orange")


anomaly_idx = df[(df["speed_anomaly"]) | (df["altitude_anomaly"]) | (df["heading_anomaly"])].index
for i in anomaly_idx:
    plt.axvline(df.loc[i, "timestamp"], color="red", linestyle="--", alpha=0.5)

plt.legend()
plt.title("Drone Telemetry & Anomalies")
plt.xlabel("Time")
plt.ylabel("Values")
plt.grid(True)
plt.tight_layout()
plt.show()