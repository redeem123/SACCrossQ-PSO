#!/usr/bin/env python3
"""
stream_gps_rest.py — Parrot Anafi USA GPS via REST API (NO Docker, native macOS)
==================================================================================
Works when the Mac is connected to the drone's WiFi directly:
  - Connect Mac WiFi to "ANAFI_USA-XXXXXX" (not the Sky Controller)
  - Drone IP:  192.168.42.1
  - OR via Sky Controller WebSocket: connect Mac WiFi to Sky Controller hotspot

Also works via Sky Controller USB if macOS recognises it as an Ethernet adapter:
  - Check System Settings → Network for a new USB Ethernet device

Usage:
    python3 stream_gps_rest.py --duration 120 --output logs/gps_track.csv
    python3 stream_gps_rest.py --ip 192.168.42.1   # direct WiFi
    python3 stream_gps_rest.py --ip 192.168.53.1   # via Sky Controller

Dependencies (already installed with requirements_macos.txt):
    pip3.10 install requests websocket-client
"""

import argparse
import csv
import json
import os
import sys
import time
import threading
from datetime import datetime

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
    sys.exit(1)

# Try websocket-client for live streaming (optional but better)
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


# ── REST API endpoints (Parrot Anafi) ────────────────────────────────────────

def get_drone_state(ip: str) -> dict:
    """
    Poll the drone's REST API for current state (GPS, battery, attitude).
    Tries multiple known Parrot API endpoint patterns.
    """
    endpoints = [
        f"http://{ip}/api/v1/drone.json",             # Media server format
        f"http://{ip}/api/v1/nrm/drone.json",         # NRM format
        f"http://{ip}/",                               # Root
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"raw": r.text[:200]}
        except requests.exceptions.ConnectionError:
            continue
        except Exception:
            continue
    return {}


def check_connection(ip: str) -> bool:
    """Verify the drone/controller is reachable."""
    try:
        r = requests.get(f"http://{ip}/", timeout=3)
        return True
    except Exception:
        return False


# ── WebSocket GPS Stream (preferred) ─────────────────────────────────────────

class WSGPSStreamer:
    """
    Stream GPS via Parrot's WebSocket (faster, event-driven, ~5Hz).
    Endpoint: ws://<drone_ip>/api/v1/SkyController2/state  (Sky Controller)
              ws://<drone_ip>/api/v1/drone                 (direct drone)
    """
    def __init__(self, ip: str, output_path: str):
        self.ip = ip
        self.output_path = output_path
        self.records = []
        self.start_time = None
        self._stop = threading.Event()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Parrot sends nested JSON; look for GPS position
            lat = lon = alt = None

            if "PositionChanged" in str(data):
                pos = data.get("ardrone3", {}).get("PilotingState", {}).get("PositionChanged", {})
                lat = pos.get("latitude")
                lon = pos.get("longitude")
                alt = pos.get("altitude")

            # Flat format
            if lat is None:
                lat = data.get("latitude") or data.get("gps_latitude")
                lon = data.get("longitude") or data.get("gps_longitude")
                alt = data.get("altitude") or data.get("gps_altitude")

            if lat is not None and lon is not None:
                t = time.time() - self.start_time
                record = {"t_sec": round(t, 2), "latitude": lat,
                          "longitude": lon, "altitude_m": alt or 0.0,
                          "timestamp": datetime.utcnow().isoformat()}
                self.records.append(record)
                print(f"[GPS] t={t:.1f}s  lat={lat:.6f}  lon={lon:.6f}  alt={alt or 0:.1f}m")
        except Exception:
            pass

    def _on_error(self, ws, error):
        print(f"[WS ERROR] {error}")

    def _on_close(self, ws, *args):
        print("[WS] Connection closed.")

    def stream(self, duration_sec: int):
        self.start_time = time.time()
        ws_urls = [
            f"ws://{self.ip}/api/v1/SkyController2/state",
            f"ws://{self.ip}/api/v1/drone",
            f"ws://{self.ip}/",
        ]
        for url in ws_urls:
            try:
                print(f"[INFO] Trying WebSocket: {url}")
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                t = threading.Thread(target=ws.run_forever, daemon=True)
                t.start()
                time.sleep(2)
                if self.records:
                    break   # This URL works
                ws.close()
            except Exception as e:
                print(f"[WARN] {url}: {e}")
                continue

        # Wait for duration
        try:
            end = time.time() + duration_sec if duration_sec > 0 else float("inf")
            while time.time() < end:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")


# ── Polling GPS Stream (fallback, no websocket-client) ───────────────────────

class PollGPSStreamer:
    """
    Poll GPS via REST at 1 Hz (fallback when WebSocket unavailable).
    """
    def __init__(self, ip: str, output_path: str):
        self.ip = ip
        self.output_path = output_path
        self.records = []
        self.start_time = None

    def stream(self, duration_sec: int):
        self.start_time = time.time()
        print(f"[INFO] Polling GPS at 1 Hz from http://{self.ip} for {duration_sec}s...")
        end = time.time() + duration_sec if duration_sec > 0 else float("inf")
        prev_key = None

        while time.time() < end:
            try:
                state = get_drone_state(self.ip)
                lat = state.get("latitude") or state.get("gps_latitude")
                lon = state.get("longitude") or state.get("gps_longitude")
                alt = state.get("altitude") or state.get("gps_altitude", 0.0)

                if lat and lon:
                    key = (round(lat, 6), round(lon, 6))
                    if key != prev_key:  # Only record when position changes
                        t = time.time() - self.start_time
                        record = {"t_sec": round(t, 2), "latitude": lat,
                                  "longitude": lon, "altitude_m": float(alt),
                                  "timestamp": datetime.utcnow().isoformat()}
                        self.records.append(record)
                        print(f"[GPS] t={t:.1f}s  lat={lat:.6f}  lon={lon:.6f}  alt={float(alt):.1f}m")
                        prev_key = key
                else:
                    print(f"[POLL] No GPS in response yet (state keys: {list(state.keys())[:5]})")

            except KeyboardInterrupt:
                print("\n[INFO] Stopped by user.")
                break
            except Exception as e:
                print(f"[WARN] Poll error: {e}")

            time.sleep(1)

    @property
    def records(self):
        return self._records

    @records.setter
    def records(self, v):
        self._records = v


def save_records(records, output_path):
    if not records:
        print("[WARN] No records to save.")
        return
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_path.replace(".csv", f"_{ts}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_sec","latitude","longitude","altitude_m","timestamp"])
        w.writeheader()
        w.writerows(records)
    print(f"[OK] Saved {len(records)} GPS points → {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Stream GPS from Parrot Anafi USA (no Docker)")
    parser.add_argument("--ip",       default="192.168.42.1",
                        help="Drone IP [default: 192.168.42.1 direct WiFi] or 192.168.53.1 Sky Ctrl")
    parser.add_argument("--output",   default="logs/gps_track.csv")
    parser.add_argument("--duration", type=int, default=60, help="Streaming duration in seconds (0=until Ctrl-C)")
    args = parser.parse_args()

    print("=" * 55)
    print(f"  GPS Streamer — Parrot Anafi USA")
    print(f"  Target: {args.ip}")
    print("=" * 55)

    # --- Connection check ---
    print(f"\n[1/3] Checking connection to {args.ip}...")
    if not check_connection(args.ip):
        print(f"\n[FAIL] Cannot reach {args.ip}")
        print("\nTo connect:")
        print("  Option A (direct WiFi — easiest):")
        print("    1. Power on the Anafi USA")
        print("    2. Go to Mac WiFi → connect to 'ANAFI_USA-XXXXXX'")
        print("    3. Re-run this script (default IP 192.168.42.1)")
        print("\n  Option B (via Sky Controller WiFi hotspot):")
        print("    1. Power on Sky Controller + Anafi USA")
        print("    2. Connect Mac WiFi to Sky Controller's hotspot SSID")
        print("    3. Run with: python3 stream_gps_rest.py --ip <skyctrl-hotspot-ip>")
        sys.exit(1)

    print(f"[OK] Connected to {args.ip}")

    # --- Quick state dump ---
    print("\n[2/3] Reading drone state...")
    state = get_drone_state(args.ip)
    if state:
        print(f"[INFO] State response: {json.dumps(state, indent=2)[:400]}")
    else:
        print("[WARN] No JSON state returned (drone may still be booting)")

    # --- Stream GPS ---
    print(f"\n[3/3] Streaming GPS for {args.duration}s...")
    if HAS_WEBSOCKET:
        print("[INFO] Using WebSocket (real-time ~5 Hz)")
        streamer = WSGPSStreamer(args.ip, args.output)
        streamer.stream(args.duration)
        records = streamer.records
    else:
        print("[INFO] websocket-client not installed — using 1Hz REST polling")
        print("[TIP] For better GPS: pip install websocket-client")
        streamer = PollGPSStreamer(args.ip, args.output)
        streamer.stream(args.duration)
        records = streamer.records

    save_records(records, args.output)


if __name__ == "__main__":
    main()
