#!/usr/bin/env python3
"""
stream_gps.py — Parrot Anafi USA GPS Streamer
==============================================
Runs inside Docker (Linux) connected to Sky Controller via USB-RNDIS.
Streams live GPS telemetry from the drone and saves to a CSV log.

Connection: Mac → USB-C → Sky Controller → WiFi → Anafi USA
Sky Controller IP: 192.168.53.1

Usage (inside Docker container):
    python stream_gps.py --output logs/gps_track.csv --duration 60

The CSV is then read by plot_realpath.py (runs natively on macOS).
"""

import olympe
import argparse
import csv
import time
import os
import json
from datetime import datetime
from olympe.messages.ardrone3.PilotingState import PositionChanged, SpeedChanged, AttitudeChanged
from olympe.messages.common.CommonState import BatteryStateChanged
from olympe.messages.ardrone3.GPSState import NumberOfSatelliteChanged

# Sky Controller RNDIS IP (USB connection from Linux/Docker)
SKYCONTROLLER_IP = "192.168.53.1"

class GPSStreamer:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.records = []
        self.drone = None
        self.start_time = None

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    def connect(self):
        """Connect to drone via Sky Controller."""
        print(f"[INFO] Connecting to Sky Controller at {SKYCONTROLLER_IP} ...")

        try:
            self.drone = olympe.SkyController3(SKYCONTROLLER_IP)
        except AttributeError:
            # Fallback: try generic Drone class through Sky Controller
            self.drone = olympe.Drone(SKYCONTROLLER_IP)

        result = self.drone.connect()
        if not result:
            raise ConnectionError(f"Failed to connect to Sky Controller at {SKYCONTROLLER_IP}.\n"
                                  "Check:\n"
                                  "  1. Sky Controller is powered ON\n"
                                  "  2. USB-C cable is plugged into Sky Controller (not the drone)\n"
                                  "  3. RNDIS network interface is up: 'ip addr show' should show 192.168.53.x")

        print("[OK] Connected to Sky Controller!")

        # Check drone state
        battery = self.drone.get_state(BatteryStateChanged)
        sats = self.drone.get_state(NumberOfSatelliteChanged)

        if battery:
            print(f"[INFO] Battery: {battery['percent']}%")
            if battery['percent'] < 20:
                print("[WARN] Battery below 20% — consider charging before flight")

        if sats:
            n_sats = sats['numberOfSatellite']
            print(f"[INFO] GPS satellites: {n_sats}")
            if n_sats < 6:
                print(f"[WARN] Only {n_sats} satellites — GPS accuracy will be poor. Wait for more satellites.")
            else:
                print(f"[OK] Good GPS fix ({n_sats} satellites)")

    def _on_position_changed(self, event, scheduler):
        """Callback: fires every time drone GPS position changes."""
        state = event.args
        timestamp = time.time() - self.start_time if self.start_time else 0

        lat  = state.get("latitude",  float("nan"))
        lon  = state.get("longitude", float("nan"))
        alt  = state.get("altitude",  float("nan"))

        record = {
            "t_sec":     round(timestamp, 2),
            "latitude":  lat,
            "longitude": lon,
            "altitude_m": alt,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.records.append(record)
        print(f"[GPS] t={timestamp:.1f}s  lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f}m")

    def stream(self, duration_sec: int):
        """Stream GPS for the given duration (seconds). 0 = stream until Ctrl-C."""
        print(f"\n[INFO] Streaming GPS for {duration_sec}s (Ctrl-C to stop early)...")
        self.start_time = time.time()

        # Subscribe to position updates
        self.drone.subscribe(self._on_position_changed, PositionChanged(_policy="wait"))

        try:
            if duration_sec > 0:
                time.sleep(duration_sec)
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Streaming stopped by user.")
        finally:
            self.drone.unsubscribe(self._on_position_changed)

        print(f"[INFO] Captured {len(self.records)} GPS records.")

    def save(self):
        """Save records to CSV."""
        if not self.records:
            print("[WARN] No GPS records to save.")
            return

        fieldnames = ["t_sec", "latitude", "longitude", "altitude_m", "timestamp"]
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[OK] GPS track saved to: {self.output_path}  ({len(self.records)} points)")

        # Also save a summary JSON
        summary_path = self.output_path.replace(".csv", "_summary.json")
        summary = {
            "total_points": len(self.records),
            "duration_sec": self.records[-1]["t_sec"] if self.records else 0,
            "lat_range": [min(r["latitude"] for r in self.records),
                          max(r["latitude"] for r in self.records)],
            "lon_range": [min(r["longitude"] for r in self.records),
                          max(r["longitude"] for r in self.records)],
            "alt_range": [min(r["altitude_m"] for r in self.records),
                          max(r["altitude_m"] for r in self.records)],
            "file": self.output_path
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary saved to: {summary_path}")

    def disconnect(self):
        if self.drone:
            self.drone.disconnect()
            print("[INFO] Disconnected from Sky Controller.")


def main():
    parser = argparse.ArgumentParser(description="Stream GPS from Parrot Anafi USA via Olympe")
    parser.add_argument("--output", default="logs/gps_track.csv",
                        help="Output CSV path (default: logs/gps_track.csv)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Streaming duration in seconds (0 = until Ctrl-C)")
    args = parser.parse_args()

    # Timestamp the output file automatically
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output.replace(".csv", f"_{ts}.csv")

    streamer = GPSStreamer(output_path=output)
    try:
        streamer.connect()
        streamer.stream(duration_sec=args.duration)
        streamer.save()
    finally:
        streamer.disconnect()


if __name__ == "__main__":
    main()
