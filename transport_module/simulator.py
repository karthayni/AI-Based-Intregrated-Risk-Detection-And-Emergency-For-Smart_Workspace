"""GPS simulator: pretends to be the GPS trackers in your vehicles, so you can demo Module 3 without a real bus.

Start the server first (python app.py), then in a second terminal:

    python simulator.py                      # demo: BUS-01 normal, BUS-02 deviates and stops, BUS-03 loses signal
    python simulator.py --all-normal         # everything goes well
    python simulator.py --vehicle BUS-02     # only one vehicle
    python simulator.py --vehicle BUS-01 --scenarios deviation stop signal

Scenarios:  deviation = leaves the route for a while | stop = stands still on the road | signal = stops sending data
Time runs faster than real life (see --speed-up) so a whole trip takes about a minute.
"""
import argparse
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import tracker as T

HERE = Path(__file__).resolve().parent
DEMO = {"BUS-01": [], "BUS-02": ["deviation", "stop"], "BUS-03": ["signal"]}   # used when routes.json has these ids


def build_path(points, step_m):
    """Walk along the route and return points every step_m metres, plus the cumulative distance of each."""
    path, cum = [tuple(points[0])], [0.0]
    carry = 0.0
    for a, b in zip(points, points[1:]):
        seg = T.haversine_m(a, b)
        d = step_m - carry
        while d <= seg:
            f = d / seg
            path.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            cum.append(cum[-1] + step_m)
            d += step_m
        carry = seg - (d - step_m)
    if path[-1] != tuple(points[-1]):
        path.append(tuple(points[-1]))
        cum.append(cum[-1] + T.haversine_m(path[-2], path[-1]))
    return path, cum


def sideways(path, i, metres):
    """Point `metres` to the left of the road at path[i], whichever direction the road runs."""
    a, b = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
    cos = math.cos(math.radians(path[i][0]))
    dx, dy = (b[1] - a[1]) * cos * 111195.0, (b[0] - a[0]) * 111195.0       # direction of travel in metres (east, north)
    n = math.hypot(dx, dy) or 1.0
    px, py = -dy / n, dx / n                                                # turned 90 degrees to the left
    return path[i][0] + metres * py / 111195.0, path[i][1] + metres * px / (111195.0 * cos)


def post(url, payload, key=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def pick_trigger(path, stops, frac, margin_m=250):
    """Index in the path near fraction `frac` of the route, moved away from any planned stop."""
    n = len(path)
    for k in range(0, 30):
        i = min(n - 2, int(n * (frac + 0.015 * k)))
        if all(T.haversine_m(path[i], (s["lat"], s["lon"])) > margin_m for s in stops):
            return i
    return int(n * frac)


def run_vehicle(v, scenarios, a, results):
    stops = v["route"]["stops"]
    path, _ = build_path(v["route"]["points"], a.speed_ms * a.tick)
    i_dev = pick_trigger(path, stops, 0.30)
    i_stop = pick_trigger(path, stops, 0.55)
    i_sig = pick_trigger(path, stops, 0.75)
    rnd = random.Random(v["id"])
    sim_t = time.time()
    url = f"http://{a.host}:{a.port}/api/location"
    n_sent = 0

    def send(lat, lon, speed):
        nonlocal sim_t, n_sent
        noise = 4 / 111195.0
        payload = dict(vehicle=v["id"], lat=lat + rnd.uniform(-noise, noise), lon=lon + rnd.uniform(-noise, noise),
                       speed=round(speed, 1), ts=sim_t, simulated=True)
        try:
            post(url, payload, a.key)
        except urllib.error.URLError as e:
            print(f"\nCannot reach the server at {url}. Is 'python app.py' running in another terminal? ({e.reason})", flush=True)
            os._exit(1)                                   # SystemExit would be ignored inside a thread
        sim_t += a.tick
        n_sent += 1
        time.sleep(a.interval)

    stop_idx = [min(range(len(path)), key=lambda i: T.haversine_m(path[i], (s["lat"], s["lon"]))) for s in stops]
    i = 0
    while i < len(path):
        lat, lon = path[i]
        if "deviation" in scenarios and i == i_dev:
            ramp = [100, 200, 300, 400, 400, 400, 400, 400, 300, 200, 100]        # metres to the side of the road
            for k, off in enumerate(ramp):
                j = min(i + k, len(path) - 1)
                send(*sideways(path, j, off), a.speed_ms)
            i = min(i + len(ramp), len(path) - 1)
            continue
        if "stop" in scenarios and i == i_stop:
            for _ in range(int(a.stop_sim_seconds / a.tick)):
                send(lat, lon, 0.0)
            i += 1
            continue
        if "signal" in scenarios and i == i_sig:
            print(f"[{v['id']}] simulating a signal loss for {a.signal_gap:.0f} s ...", flush=True)
            time.sleep(a.signal_gap)
            sim_t += a.signal_gap * (a.tick / a.interval)                         # time kept passing for the vehicle
            i += 1
            continue
        send(lat, lon, a.speed_ms)
        if i in stop_idx and i != len(path) - 1:                                  # wait at a planned stop (pickup)
            for _ in range(int(a.dwell_sim_seconds / a.tick)):
                send(lat, lon, 0.0)
        i += 1
    for _ in range(4):                                                            # parked at the destination
        send(*path[-1], 0.0)
    results[v["id"]] = n_sent
    print(f"[{v['id']}] finished ({n_sent} readings sent)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="GPS simulator for Module 3")
    ap.add_argument("--config", default=str(HERE / "routes.json"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--key", default=None, help="API key if the server was started with --key")
    ap.add_argument("--vehicle", default=None, help="simulate only this vehicle id")
    ap.add_argument("--scenarios", nargs="*", default=None, choices=["deviation", "stop", "signal"],
                    help="problems to simulate for the chosen vehicle(s)")
    ap.add_argument("--all-normal", action="store_true")
    ap.add_argument("--speed-ms", type=float, default=8.0, help="vehicle speed in metres per second (8 = 29 km/h)")
    ap.add_argument("--tick", type=float, default=10.0, help="simulated seconds between GPS readings")
    ap.add_argument("--interval", type=float, default=0.4, help="real seconds between readings (smaller = faster demo)")
    ap.add_argument("--dwell-sim-seconds", type=float, default=40, help="simulated waiting time at each pickup")
    ap.add_argument("--stop-sim-seconds", type=float, default=300, help="simulated length of the unexpected stop")
    ap.add_argument("--signal-gap", type=float, default=40, help="real seconds without any readings in the signal scenario (must be longer than signal_timeout_s)")
    a = ap.parse_args()

    cfg = T.load_config(a.config)
    vehicles = [v for v in cfg["vehicles"] if a.vehicle in (None, v["id"])]
    if not vehicles:
        raise SystemExit(f"No vehicle '{a.vehicle}' in {a.config}")
    results, threads = {}, []
    for v in vehicles:
        if a.all_normal:
            sc = []
        elif a.scenarios is not None:
            sc = a.scenarios
        else:
            sc = DEMO.get(v["id"], [])
        print(f"[{v['id']}] starting, scenarios: {', '.join(sc) or 'none (normal trip)'}", flush=True)
        t = threading.Thread(target=run_vehicle, args=(v, sc, a, results), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
