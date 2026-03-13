# Parrot Anafi USA — Connection & Telemetry Notes

Documented from live testing session: 2026-02-23

---

## Hardware Setup

- **Drone**: Parrot Anafi USA
- **Host**: macOS (Apple Silicon)
- **Connection**: Mac WiFi → `ANAFI_USA-XXXXXX` (drone's own WiFi hotspot)
- **Internet**: iPhone USB tethering (via Lightning/USB-C) simultaneously

> **No Sky Controller needed.** No Docker. No Linux. No USB. Pure Python 3.10 on macOS.

---

## Network Topology

```
[Parrot Anafi USA]
    WiFi SSID: ANAFI_USA-XXXXXX
    IP: 192.168.42.1

[Mac]
    WiFi interface (en0): 192.168.42.x  ← drone commands/telemetry
    USB interface (en13): DHCP from iPhone ← internet access

[iPhone]
    Personal Hotspot (USB) → Mac internet
```

Both interfaces work **simultaneously** — drone WiFi does not block internet.

---

## What Did NOT Work

| Approach | Why it failed |
|---|---|
| Sky Controller USB → Mac (RNDIS) | macOS removed RNDIS support; no `192.168.53.x` interface appears |
| Docker (Olympe) | Docker not installed; even if installed, macOS Docker Desktop can't bridge USB RNDIS to containers |
| Parrot REST API (`/api/v1/...`) | All endpoints return 404 on Anafi USA firmware |
| UDP discovery (port 44444) | Anafi USA uses TCP, not UDP, for ARSDK3 discovery |
| WebSocket (`ws://192.168.42.1/api/v1/...`) | 404 — websocket endpoints don't exist in this firmware |

---

## What DOES Work — ARSDK3 Protocol

The Anafi USA uses **ARSDK3** (Parrot's binary protocol). Connection sequence:

### Step 1: TCP Discovery (port 44444)

Send a JSON `\0`-terminated string:
```json
{"controller_name": "MacOS", "controller_type": "computer", "d2c_port": 43210}
```

Drone responds:
```json
{"c2d_update_port": 51, "c2d_user_port": 21, "status": 0, "c2d_port": 2233, "qos_mode": 0, "proto_v": 1}
```

Key fields:
- `c2d_port: 2233` — send commands **to** drone on UDP port 2233
- `d2c_port: 43210` — listen for telemetry **from** drone on UDP port 43210

### Step 2: Heartbeat (keep session alive)

Send an empty ARSDK3 ACK packet to `192.168.42.1:2233` every 1 second:
```python
import struct
heartbeat = struct.pack('<BBBI', 2, 0, 0, 7)  # type=2, buf=0, seq=0, size=7
```
Without heartbeats the drone stops sending telemetry within ~5 seconds.

### Step 3: Receive Telemetry on UDP 43210

ARSDK3 packet structure:
```
Byte 0    : data_type
Byte 1    : buffer_id
Byte 2    : sequence_number
Bytes 3-6 : total_size (little-endian uint32)
Bytes 7+  : payload
```

Payload structure:
```
Byte 0   : project_id
Byte 1   : class_id
Bytes 2-3: command_id (little-endian uint16)
Bytes 4+ : command arguments
```

---

## Available Telemetry Commands (confirmed live)

| project | class | cmd | Data | Arg format |
|---|---|---|---|---|
| 1 | 4 | 4 | **GPS position** | `double lat, lon, alt` |
| 1 | 4 | 5 | Speed (NED) | `float vN, vE, vDown` |
| 1 | 4 | 6 | Altitude (baro) | `float alt` |
| 1 | 4 | 8 | Attitude | `float roll, pitch, yaw` |
| 1 | 4 | 9 | GPS satellite info | mixed |
| 1 | 4 | 21 | GNSS satellite count | `uint8 count` |
| 148 | 0 | 6 | Battery / system state | mixed |
| 149 | 0 | 3/7/9 | Camera/gimbal state | mixed |
| 143 | 0 | 10/22 | Unknown (possibly video) | — |

### GPS Sentinel Value
When **no GPS fix**, the drone sends `lat=500.0, lon=500.0, alt=500.0`.
This means the drone is indoors or has no sky view.
**Always place drone outdoors and wait ~30s before expecting real coordinates.**

---

## GPS Data Range (from live test)

Drone was tested **indoors** — all GPS returned sentinel values:
```
[NO-FIX] t=0.0s  lat=500.0000000  lon=500.0000000  alt=500.0m
```

When outdoors with fix, expected output (Hanoi area):
```
[FIX] t=0.0s  lat=21.005XXX  lon=105.845XXX  alt=14.2m
```

---

## Python Code — Minimal GPS Streamer

```python
import socket, json, struct, time, threading

DRONE_IP, DISC_PORT, D2C_PORT = "192.168.42.1", 44444, 43210

# Discover
disc = json.dumps({"controller_name":"MacOS","controller_type":"computer","d2c_port":D2C_PORT}).encode() + b'\x00'
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((DRONE_IP, DISC_PORT)); s.sendall(disc)
resp = json.loads(s.recv(4096).decode().strip('\x00')); s.close()
C2D_PORT = resp['c2d_port']   # 2233

# Heartbeat thread
def hb():
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pkt = struct.pack('<BBBI', 2, 0, 0, 7)
    while True:
        u.sendto(pkt, (DRONE_IP, C2D_PORT)); time.sleep(1)
threading.Thread(target=hb, daemon=True).start()

# Listen for GPS
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind(('', D2C_PORT)); udp.settimeout(1)
while True:
    try:
        data, _ = udp.recvfrom(65535)
        if len(data) < 11: continue
        p = data[7:]
        if p[0]==1 and p[1]==4 and struct.unpack_from('<H',p,2)[0]==4 and len(p)>=28:
            lat, lon, alt = struct.unpack_from('<ddd', p, 4)
            if abs(lat) < 400:
                print(f"GPS: {lat:.7f}, {lon:.7f}, {alt:.1f}m")
    except socket.timeout: pass
```

---

## Scripts in This Folder

| File | Purpose | Requires |
|---|---|---|
| `connect_test.py` | Hardware check (needs Olympe/Docker) | Docker |
| `stream_gps.py` | GPS stream via Olympe (Linux/Docker) | Docker |
| `stream_gps_rest.py` | GPS via REST/WebSocket (outdated) | Python 3.10 |
| `plot_realpath.py` | Plot GPS CSV on terrain map | Python 3.10 + rasterio |
| `/tmp/gps_decode.py` | **Working ARSDK3 GPS streamer (macOS)** | Python 3.10 only |

> **Best script to use**: `/tmp/gps_decode.py` — no dependencies beyond stdlib. Copy it to this folder as `stream_gps_arsdk3.py` for permanent use.

---

## Next Steps

1. **Outdoor GPS test**: place drone outside → wait 30s → run `gps_decode.py`
2. **Record flight path**: stream GPS during manual flight via Sky Controller
3. **Pull video**: decode ARSDK3 video stream (proj=143 packets) with OpenCV
4. **Send commands**: use c2d_port 2233 to send autonomous waypoints
