import socket, json, struct, time, csv, os, threading
from datetime import datetime

DRONE_IP  = "192.168.42.1"
DISC_PORT = 44444
D2C_PORT  = 43210
C2D_PORT  = 2233
DURATION  = 30

# Discovery
disc = json.dumps({"controller_name":"MacOS","controller_type":"computer","d2c_port":D2C_PORT}).encode() + b'\x00'
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5); s.connect((DRONE_IP, DISC_PORT)); s.sendall(disc)
resp = s.recv(4096); s.close()
print("Discovery OK")

# Heartbeat thread
def hb():
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pkt = struct.pack('<BBBI', 2, 0, 0, 7)
    for _ in range(DURATION + 5):
        u.sendto(pkt, (DRONE_IP, C2D_PORT)); time.sleep(1)
    u.close()
threading.Thread(target=hb, daemon=True).start()

# Listen + decode
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
udp.bind(('', D2C_PORT)); udp.settimeout(0.5)
print(f"Listening {DURATION}s for GPS (proj=1 cls=4 cmd=4) ...")
records = []; end = time.time() + DURATION; t0 = time.time()

while time.time() < end:
    try:
        data, addr = udp.recvfrom(65535)
        if len(data) < 11: continue
        payload = data[7:]
        proj = payload[0]; cls = payload[1]
        cmd  = struct.unpack_from('<H', payload, 2)[0]
        args = payload[4:]
        if proj == 1 and cls == 4 and cmd == 4 and len(args) >= 24:
            lat, lon, alt = struct.unpack_from('<ddd', args)
            t = round(time.time() - t0, 2)
            # 500.0 = Parrot no-fix sentinel
            fix = abs(lat) < 400 and abs(lon) < 400
            status = "FIX" if fix else "NO-FIX(indoor?)"
            print(f"[{status}] t={t}s  lat={lat:.7f}  lon={lon:.7f}  alt={alt:.1f}m")
            if fix:
                records.append({"t_sec":t,"latitude":lat,"longitude":lon,
                                "altitude_m":round(alt,2),
                                "timestamp":datetime.utcnow().isoformat()})
    except socket.timeout: pass
udp.close()

if records:
    os.makedirs("validation/anafi/logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"validation/anafi/logs/gps_track_{ts}.csv"
    with open(out,'w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=["t_sec","latitude","longitude","altitude_m","timestamp"])
        w.writeheader(); w.writerows(records)
    print(f"\nSaved {len(records)} GPS pts -> {out}")
else:
    print("\nGPS fix not acquired — drone needs to be outdoors!")
    print("Move drone outside, wait 30s for fix, then re-run.")
