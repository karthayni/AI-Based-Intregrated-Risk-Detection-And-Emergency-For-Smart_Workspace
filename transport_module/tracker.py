"""Module 3: employee transport tracking logic.

This file has NO web code. It receives GPS points, compares them with the planned route and raises alerts:

  ROUTE_DEVIATION  vehicle is further than `deviation_m` from the planned route for a few readings in a row
  UNEXPECTED_STOP  vehicle stayed in one place for `stop_seconds` and is not at a planned stop
  SIGNAL_LOST      no GPS reading received for `signal_timeout_s` seconds
  STOP_MISSED      vehicle reached a later stop without visiting an earlier one

and keeps the pickup / drop status of every stop:  pending -> arrived -> done  (or missed).

Everything is stored in a small SQLite file (transport.db) with the tables from the ER diagram:
vehicle, gps_log, transport_alert, stop_event.
"""
import json
import math
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

EARTH_R = 6371000.0

DEFAULTS = {
    "deviation_m": 150,        # metres away from the route that counts as "off route"
    "deviation_pings": 3,      # ...for this many readings in a row (ignores one noisy reading)
    "stop_seconds": 120,       # standing still this long, away from any stop = unexpected stop
    "still_radius_m": 25,      # moving less than this counts as "standing still" (GPS noise is about 5-15 m)
    "stop_radius_m": 60,       # being this close to a planned stop = "at the stop"
    "signal_timeout_s": 60,    # seconds without a reading before SIGNAL_LOST
    "max_speed_ms": 60,        # a jump faster than this (216 km/h) is treated as a GPS glitch and ignored
}

SEVERITY = {"ROUTE_DEVIATION": "high", "UNEXPECTED_STOP": "medium", "SIGNAL_LOST": "medium", "STOP_MISSED": "medium"}


# ----------------------------------------------------------------------------- geometry
def haversine_m(a, b):
    """Distance in metres between two (lat, lon) points."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def _xy(p, lat0):
    """Lat/lon -> flat x,y in metres around latitude lat0 (accurate enough over a city)."""
    return (EARTH_R * math.radians(p[1]) * math.cos(math.radians(lat0)), EARTH_R * math.radians(p[0]))


def point_to_segment_m(p, a, b):
    """Shortest distance in metres from point p to the line segment a-b."""
    lat0 = p[0]
    px, py = _xy(p, lat0)
    ax, ay = _xy(a, lat0)
    bx, by = _xy(b, lat0)
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_route_m(p, points):
    """Distance in metres from p to the nearest part of the route (a list of (lat, lon))."""
    return min(point_to_segment_m(p, points[i], points[i + 1]) for i in range(len(points) - 1))


# ----------------------------------------------------------------------------- config
def load_config(path):
    """Read routes.json and check it, with clear messages if something is wrong."""
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")
    vehicles = cfg.get("vehicles")
    if not vehicles:
        raise SystemExit(f"{path}: needs a non-empty 'vehicles' list")
    seen = set()
    for v in vehicles:
        vid = v.get("id")
        if not vid or vid in seen:
            raise SystemExit(f"{path}: every vehicle needs a unique 'id' (problem near {vid!r})")
        seen.add(vid)
        r = v.get("route") or {}
        pts = r.get("points") or []
        if len(pts) < 2 or any(len(p) != 2 for p in pts):
            raise SystemExit(f"{path}: vehicle {vid} needs route.points with at least 2 [lat, lon] pairs")
        for s in r.get("stops") or []:
            if not {"name", "lat", "lon"} <= set(s):
                raise SystemExit(f"{path}: vehicle {vid} has a stop without name/lat/lon: {s}")
        if not r.get("stops"):
            raise SystemExit(f"{path}: vehicle {vid} needs at least one stop in route.stops")
    cfg["settings"] = {**DEFAULTS, **cfg.get("settings", {})}
    return cfg


# ----------------------------------------------------------------------------- one vehicle
class Vehicle:
    def __init__(self, cfg):
        self.id = cfg["id"]
        self.name = cfg.get("name", cfg["id"])
        self.driver = cfg.get("driver", "")
        r = cfg["route"]
        self.route_name = r.get("name", "")
        self.points = [(float(a), float(b)) for a, b in r["points"]]
        self.stops = [dict(name=s["name"], lat=float(s["lat"]), lon=float(s["lon"]), kind=s.get("kind", "pickup"))
                      for s in r["stops"]]
        self.reset()

    def reset(self):
        self.trip = "not_started"             # not_started | in_progress | completed
        self.last = None                      # dict(ts, lat, lon)
        self.last_rx = None                   # wall-clock time of the last reading
        self.speed_ms = None
        self.off_route_m = None
        self.off_count = self.back_count = 0
        self.anchor = None                    # (lat, lon, ts) where the vehicle started standing still
        self.trail = deque(maxlen=150)
        self.active = {}                      # alert type -> alert dict (alerts that are still open)
        for s in self.stops:
            s["status"], s["arrived_ts"], s["done_ts"], s["arrived_at"], s["done_at"] = "pending", None, None, None, None


# ----------------------------------------------------------------------------- tracker
class Tracker:
    def __init__(self, config, db_path="transport.db", notify=None):
        self.cfg = config
        self.s = config["settings"]
        self.vehicles = {v["id"]: Vehicle(v) for v in config["vehicles"]}
        self.notify = notify or (lambda alert: None)
        self.lock = threading.RLock()
        self._wall = time.time()               # real clock of the reading being processed (for display)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS vehicle(vehicle_id TEXT PRIMARY KEY, vehicle_no TEXT, route_name TEXT, driver_name TEXT);
            CREATE TABLE IF NOT EXISTS gps_log(log_id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT, ts REAL,
                latitude REAL, longitude REAL, speed_ms REAL, off_route_m REAL);
            CREATE TABLE IF NOT EXISTS transport_alert(alert_id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT,
                alert_type TEXT, severity TEXT, message TEXT, latitude REAL, longitude REAL, started_ts REAL, resolved_ts REAL, created_at REAL);
            CREATE TABLE IF NOT EXISTS stop_event(event_id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT,
                stop_name TEXT, status TEXT, ts REAL);
        """)
        for v in self.vehicles.values():
            self.db.execute("INSERT OR REPLACE INTO vehicle VALUES (?,?,?,?)", (v.id, v.name, v.route_name, v.driver))
        self.db.commit()

    # ---- alerts ---------------------------------------------------------------------------------
    def _raise(self, v, kind, message, pos, ts, event=False):
        cur = self.db.execute(
            "INSERT INTO transport_alert(vehicle_id,alert_type,severity,message,latitude,longitude,started_ts,resolved_ts,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (v.id, kind, SEVERITY[kind], message, pos[0], pos[1], ts, ts if event else None, self._wall))
        self.db.commit()
        alert = dict(id=cur.lastrowid, vehicle_id=v.id, type=kind, severity=SEVERITY[kind], message=message,
                     lat=pos[0], lon=pos[1], started=ts, resolved=ts if event else None, created=self._wall)
        if not event:
            v.active[kind] = alert
        self.notify(alert)
        return alert

    def _resolve(self, v, kind, ts):
        a = v.active.pop(kind, None)
        if a:
            self.db.execute("UPDATE transport_alert SET resolved_ts=? WHERE alert_id=?", (ts, a["id"]))
            self.db.commit()

    # ---- main entry: one GPS reading --------------------------------------------------------------
    def ingest(self, vehicle_id, lat, lon, ts=None, speed=None, rx=None):
        """Process one GPS reading. Returns {'accepted': bool, 'reason': str}."""
        with self.lock:
            v = self.vehicles.get(vehicle_id)
            if v is None:
                raise KeyError(f"unknown vehicle '{vehicle_id}'")
            lat, lon = float(lat), float(lon)
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("latitude/longitude out of range")
            rx = time.time() if rx is None else rx
            self._wall = rx
            ts = rx if ts is None else float(ts)
            p = (lat, lon)

            if v.last is not None:
                dt = ts - v.last["ts"]
                if dt <= 0:
                    return dict(accepted=False, reason="old or duplicate reading")
                jump = haversine_m((v.last["lat"], v.last["lon"]), p) / max(dt, 1.0)   # readings closer than 1 s count as 1 s
                if jump > self.s["max_speed_ms"]:
                    v.last_rx = rx                       # the device is alive, but the position is nonsense
                    return dict(accepted=False, reason="GPS jump ignored")
                derived = jump
            else:
                derived = None
            v.speed_ms = float(speed) if speed is not None else derived
            v.last, v.last_rx = dict(ts=ts, lat=lat, lon=lon), rx
            v.trail.append((lat, lon))
            if v.trip == "not_started":
                v.trip = "in_progress"

            self._resolve(v, "SIGNAL_LOST", ts)
            v.off_route_m = distance_to_route_m(p, v.points)
            if v.trip == "in_progress":
                self._check_route(v, p, ts)
                self._check_stops(v, p, ts)
                if v.trip == "in_progress":
                    self._check_standing_still(v, p, ts)
            self.db.execute("INSERT INTO gps_log(vehicle_id,ts,latitude,longitude,speed_ms,off_route_m) VALUES (?,?,?,?,?,?)",
                            (v.id, ts, lat, lon, v.speed_ms, v.off_route_m))
            self.db.commit()
            return dict(accepted=True, reason="ok")

    # ---- the three checks -----------------------------------------------------------------------
    def _check_route(self, v, p, ts):
        d, lim = v.off_route_m, self.s["deviation_m"]
        if d > lim:
            v.off_count += 1
            v.back_count = 0
            if v.off_count >= self.s["deviation_pings"] and "ROUTE_DEVIATION" not in v.active:
                self._raise(v, "ROUTE_DEVIATION", f"{v.name} is {d:.0f} m away from its planned route", p, ts)
        elif d < lim * 0.7:                              # clearly back on route (the gap avoids flicker)
            v.back_count += 1
            v.off_count = 0
            if v.back_count >= 2:
                self._resolve(v, "ROUTE_DEVIATION", ts)
        elif "ROUTE_DEVIATION" not in v.active:
            v.off_count = 0

    def _check_stops(self, v, p, ts):
        r = self.s["stop_radius_m"]
        for i, st in enumerate(v.stops):
            inside = haversine_m(p, (st["lat"], st["lon"])) <= r
            if st["status"] == "pending" and inside:
                for j in range(i):                       # visiting stop i means earlier waiting stops were skipped
                    e = v.stops[j]
                    if e["status"] == "pending":
                        e["status"] = "missed"
                        self.db.execute("INSERT INTO stop_event(vehicle_id,stop_name,status,ts) VALUES (?,?,?,?)", (v.id, e["name"], "missed", ts))
                        self._raise(v, "STOP_MISSED", f"{v.name} skipped the stop '{e['name']}'", p, ts, event=True)
                st["status"], st["arrived_ts"], st["arrived_at"] = "arrived", ts, self._wall
                self.db.execute("INSERT INTO stop_event(vehicle_id,stop_name,status,ts) VALUES (?,?,?,?)", (v.id, st["name"], "arrived", ts))
                if i == len(v.stops) - 1:                # last stop reached: the trip is over
                    st["status"], st["done_ts"], st["done_at"] = "done", ts, self._wall
                    v.trip = "completed"
                    for kind in list(v.active):
                        self._resolve(v, kind, ts)
            elif st["status"] == "arrived" and not inside:
                st["status"], st["done_ts"], st["done_at"] = "done", ts, self._wall
                self.db.execute("INSERT INTO stop_event(vehicle_id,stop_name,status,ts) VALUES (?,?,?,?)", (v.id, st["name"], "done", ts))

    def _check_standing_still(self, v, p, ts):
        if v.anchor is None or haversine_m((v.anchor[0], v.anchor[1]), p) > self.s["still_radius_m"]:
            v.anchor = (p[0], p[1], ts)                  # vehicle moved: start timing again
            self._resolve(v, "UNEXPECTED_STOP", ts)
            return
        waited = ts - v.anchor[2]
        at_planned_place = any(haversine_m(p, (s["lat"], s["lon"])) <= self.s["stop_radius_m"] for s in v.stops) \
            or haversine_m(p, v.points[0]) <= self.s["stop_radius_m"]
        if waited >= self.s["stop_seconds"] and not at_planned_place and "UNEXPECTED_STOP" not in v.active:
            self._raise(v, "UNEXPECTED_STOP", f"{v.name} has been stopped for {waited/60:.1f} min away from any planned stop", p, ts)

    # ---- signal check (called on a timer, uses the real clock) ------------------------------------
    def check_signal(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self._wall = now
            for v in self.vehicles.values():
                if v.trip == "in_progress" and v.last_rx and now - v.last_rx > self.s["signal_timeout_s"] \
                        and "SIGNAL_LOST" not in v.active:
                    pos = (v.last["lat"], v.last["lon"])
                    self._raise(v, "SIGNAL_LOST", f"No GPS signal from {v.name} for {now - v.last_rx:.0f} s", pos, v.last["ts"] + (now - v.last_rx))

    def reset(self, vehicle_id):
        with self.lock:
            v = self.vehicles[vehicle_id]
            for kind in list(v.active):
                self._resolve(v, kind, time.time())
            v.reset()

    # ---- data for the dashboard -------------------------------------------------------------------
    @staticmethod
    def _label(v):
        if "SIGNAL_LOST" in v.active:
            return "NO SIGNAL"
        if "ROUTE_DEVIATION" in v.active:
            return "OFF ROUTE"
        if "UNEXPECTED_STOP" in v.active:
            return "STOPPED"
        return {"not_started": "WAITING", "in_progress": "ON ROUTE", "completed": "COMPLETED"}[v.trip]

    def snapshot(self, now=None, recent=40):
        now = time.time() if now is None else now
        with self.lock:
            out = []
            for v in self.vehicles.values():
                pos = [v.last["lat"], v.last["lon"]] if v.last else None
                out.append(dict(
                    id=v.id, name=v.name, driver=v.driver, route_name=v.route_name, status=self._label(v), trip=v.trip,
                    position=pos, speed_kmh=None if v.speed_ms is None else round(v.speed_ms * 3.6, 1),
                    off_route_m=None if v.off_route_m is None else round(v.off_route_m),
                    seconds_since_update=None if v.last_rx is None else round(now - v.last_rx),
                    points=[list(p) for p in v.points], stops=[dict(s) for s in v.stops],
                    trail=[list(t) for t in v.trail], active_alerts=list(v.active.values())))
            rows = self.db.execute("SELECT alert_id,vehicle_id,alert_type,severity,message,latitude,longitude,started_ts,resolved_ts,created_at "
                                   "FROM transport_alert ORDER BY alert_id DESC LIMIT ?", (recent,)).fetchall()
            alerts = [dict(id=r[0], vehicle_id=r[1], type=r[2], severity=r[3], message=r[4], lat=r[5], lon=r[6],
                           started=r[7], resolved=r[8], created=r[9]) for r in rows]
            return dict(now=now, vehicles=out, alerts=alerts, settings=self.s)
