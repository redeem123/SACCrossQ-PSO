# Parrot Anafi USA — GPS Validation

Stream live GPS from the Parrot Anafi USA and overlay the real flight path on the AFSACPSO simulation terrain map.

## Folder Structure

```
validation/anafi/
├── README.md               ← this file
├── connect_test.py         ← verify connectivity first
├── stream_gps.py           ← stream live GPS → saves CSV (requires Docker)
├── plot_realpath.py        ← plot GPS path on terrain map (native macOS)
├── run_docker.sh           ← builds & runs the Olympe container
├── requirements_macos.txt  ← pip packages for the plotter (no Docker needed)
├── docker/
│   └── Dockerfile          ← Ubuntu 20.04 + Parrot Olympe
└── logs/                   ← GPS CSV files saved here (git-ignored)
```

---

## Prerequisites

### Hardware
- Parrot Anafi USA (powered on, all lights solid)
- Sky Controller 3 or 4 (paired with the drone)
- USB-C cable: **Mac → Sky Controller** (not to the drone)

### Software — GPS Streamer (Docker, one-time setup)
```bash
# Install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/
# Then build the container (takes ~3 min, one-time):
bash run_docker.sh echo "Build complete"
```

### Software — Map Plotter (native macOS, instant)
```bash
pip install -r requirements_macos.txt
```

---

## Workflow

### Step 1 — Connect & verify
Plug USB-C cable into the **Sky Controller** (not the drone).
```bash
bash run_docker.sh python connect_test.py
```
Expected output:
```
[OK] Connected to Sky Controller
[OK] Battery: 82% — OK
[OK] Satellites: 9 — Good fix
[OK] Current position: lat=21.005123  lon=105.845456  alt=14.2m
```

### Step 2 — Stream GPS
```bash
# Stream for 5 minutes (300 sec):
bash run_docker.sh python stream_gps.py --duration 300

# Or stream until Ctrl-C:
bash run_docker.sh python stream_gps.py
```
GPS track saved to `logs/gps_track_YYYYMMDD_HHMMSS.csv`

### Step 3 — Plot on terrain map
```bash
python plot_realpath.py \
  --gps_csv  logs/gps_track_20260223_120000.csv \
  --origin_lat 21.005000 \
  --origin_lon 105.845000 \
  --terrain_mat ../../shared/environment/ChrismasTerrain2.mat \
  --output    ../../outputs/figures/paths/real_anafi_path.png
```

The figure shows:
- **Left**: top-down terrain contour + real GPS path (coloured by time)
- **Right**: 3D terrain surface + real GPS path (coloured by altitude)

---

## Notes

- `logs/` is git-ignored (GPS data is large and location-sensitive)
- `origin_lat/lon` = the real-world GPS coordinates that correspond to map position `[0, 0]`
- `map_scale` = metres per map unit (default 1.0 — adjust if your map is scaled differently)
- The plotter works even without flying — put the drone on the ground and stream to test connectivity
