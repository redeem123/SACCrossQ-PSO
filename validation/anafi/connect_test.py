#!/usr/bin/env python3
"""
connect_test.py — Quick connectivity check for Parrot Anafi USA via Sky Controller
Run this first to verify hardware is properly connected before streaming GPS.

Usage (inside Docker):
    python connect_test.py
"""

import olympe
import sys
import time
from olympe.messages.ardrone3.PilotingState import PositionChanged
from olympe.messages.common.CommonState import BatteryStateChanged
from olympe.messages.ardrone3.GPSState import NumberOfSatelliteChanged
from olympe.messages.common.SettingsState import ProductVersionChanged

SKYCONTROLLER_IP = "192.168.53.1"

def run_test():
    print("=" * 55)
    print("  Parrot Anafi USA — Connection Test")
    print("=" * 55)

    # 1. Connect
    print(f"\n[1/4] Connecting to Sky Controller ({SKYCONTROLLER_IP})...")
    try:
        drone = olympe.SkyController3(SKYCONTROLLER_IP)
    except AttributeError:
        drone = olympe.Drone(SKYCONTROLLER_IP)

    result = drone.connect()
    if not result:
        print("[FAIL] Cannot connect. Check:")
        print("  - Sky Controller is powered ON")
        print("  - USB-C cable is connected to Sky Controller")
        print("  - Run 'ip addr' and check 192.168.53.x network is up")
        sys.exit(1)

    print("[OK] Connected to Sky Controller")

    # 2. Firmware
    print("\n[2/4] Checking firmware...")
    try:
        fw = drone.get_state(ProductVersionChanged)
        if fw:
            print(f"[OK] Firmware: {fw.get('software', 'unknown')}")
    except Exception:
        print("[INFO] Could not read firmware version (non-critical)")

    # 3. Battery
    print("\n[3/4] Battery level...")
    bat = drone.get_state(BatteryStateChanged)
    if bat:
        pct = bat['percent']
        status = "OK" if pct >= 30 else "LOW — charge before flight"
        print(f"[{'OK' if pct >= 30 else 'WARN'}] Battery: {pct}% — {status}")
    else:
        print("[INFO] Battery state not yet available")

    # 4. GPS
    print("\n[4/4] GPS fix...")
    sats = drone.get_state(NumberOfSatelliteChanged)
    pos  = drone.get_state(PositionChanged)

    if sats:
        n = sats['numberOfSatellite']
        ok = n >= 6
        print(f"[{'OK' if ok else 'WARN'}] Satellites: {n} — {'Good fix' if ok else 'Poor fix (wait for more)'}")
    else:
        print("[INFO] Satellite count not yet available (drone may still be acquiring)")

    if pos:
        lat = pos.get('latitude',  float('nan'))
        lon = pos.get('longitude', float('nan'))
        alt = pos.get('altitude',  float('nan'))
        print(f"[OK] Current position: lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f}m")
    else:
        print("[INFO] Position not yet available — drone may be indoors or acquiring GPS")

    print("\n" + "=" * 55)
    print("  All checks done. Ready to run stream_gps.py")
    print("=" * 55 + "\n")

    drone.disconnect()

if __name__ == "__main__":
    run_test()
