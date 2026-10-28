# Unified Dashboard – Modules 1+2 (workplace) and Module 3 (transport)

Live map of employee buses with automatic alerts: ROUTE_DEVIATION, UNEXPECTED_STOP, SIGNAL_LOST, STOP_MISSED.
Stack: Python Flask + SQLite + Leaflet map (bundled, works offline except map tiles).

## Run
1. `pip install -r requirements.txt`
2. Terminal 1: `python app.py --fresh`  → open http://127.0.0.1:5000
3. Terminal 2: `python simulator.py`   (3 demo buses; one deviates, one stops, one loses signal)
4. Tests: `python -m unittest test_tracker`

## Real GPS
Send `GET/POST /api/location?id=BUS-01&lat=12.93&lon=77.59` (e.g. from a phone app such as GPSLogger).
Start the server with `--host 0.0.0.0` so a phone on the same Wi-Fi can reach it. Edit `routes.json` to change buses, routes and stops.

## Link with Modules 1+2
`python app.py --ppe-log C:\path\to\ppe_project\logs\events.csv` shows PPE/stress events on the same dashboard
(overall status, workplace tiles, combined alert timeline, snapshot pictures). Without the option it looks for `logs/events.csv` next to app.py
and in `../ppe_project` and `../ppe_module`.
Export tables: `/export/alerts.csv`.

## Website pages
`/` Home, `/about` About, and the dashboard inside the website:
`/dashboard` Overview, `/dashboard/camera` Live camera, `/dashboard/workplace` Workplace alerts, `/dashboard/transport` Transport.

## Live camera page (Modules 1+2 from the website)
Start the server with your PPE folder and the page `/dashboard/camera` can start and stop `ppe_detect.py` for you and show its live picture:
`python app.py --ppe-dir "C:\Users\karth\OneDrive\Desktop\ppe_project"`
The folder needs best.pt, ppe_detect.py (new version) and, for stress, stress.py, model_unquant.tflite and labels.txt.
The detection runs with the Python of that folder's `venv` (change with --ppe-python). Start/stop only works from the same computer.
