"""Tests for tracker.py. Run with:   python -m unittest -v test_tracker
(Uses an in-memory database, so nothing is written to disk.)"""
import math
import unittest

import tracker as T

LAT0, LON = 12.9000, 77.6000
M = 1 / 111195.0                      # one metre in degrees of latitude


def north(m):                         # a point m metres north of the start
    return (LAT0 + m * M, LON)


def east(p, m):                       # a point m metres east of p
    return (p[0], p[1] + m * M / math.cos(math.radians(p[0])))


CONFIG = {
    "settings": dict(T.DEFAULTS),
    "vehicles": [{
        "id": "BUS-1", "name": "Bus 1", "driver": "Test",
        "route": {
            "name": "Test route",
            "points": [list(north(0)), list(north(2200))],
            "stops": [
                {"name": "Stop A", "lat": north(550)[0], "lon": LON},
                {"name": "Stop B", "lat": north(1100)[0], "lon": LON},
                {"name": "Office", "lat": north(2200)[0], "lon": LON, "kind": "drop"},
            ]}}]}


def make(notify=None):
    return T.Tracker(CONFIG, db_path=":memory:", notify=notify)


def drive(tr, start_m, end_m, t0, step_m=50, dt=5, offset_east=0):
    """Drive north from start_m to end_m. Returns the time after the last reading."""
    t, m = t0, start_m
    while m <= end_m:
        p = north(m)
        if offset_east:
            p = east(p, offset_east)
        tr.ingest("BUS-1", p[0], p[1], ts=t)
        t += dt
        m += step_m
    return t


def kinds(tr):
    return [a["type"] for a in tr.vehicles["BUS-1"].active.values()]


class Geometry(unittest.TestCase):
    def test_haversine(self):
        self.assertAlmostEqual(T.haversine_m(north(0), north(1000)), 1000, delta=2)

    def test_distance_to_route(self):
        pts = [north(0), north(2000)]
        self.assertAlmostEqual(T.distance_to_route_m(east(north(1000), 100), pts), 100, delta=2)
        self.assertAlmostEqual(T.distance_to_route_m(north(-100), pts), 100, delta=2)   # before the start
        self.assertAlmostEqual(T.distance_to_route_m(north(500), pts), 0, delta=1)

    def test_degenerate_segment(self):
        self.assertAlmostEqual(T.distance_to_route_m(north(100), [north(0), north(0)]), 100, delta=2)


class NormalTrip(unittest.TestCase):
    def test_full_trip_has_no_alerts_and_updates_stops(self):
        got = []
        tr = make(notify=got.append)
        # dwell 40 s at each stop like a real pickup
        t = drive(tr, 0, 500, 1000)
        for k in range(8):
            tr.ingest("BUS-1", *north(550), ts=t); t += 5
        t = drive(tr, 600, 1050, t)
        for k in range(8):
            tr.ingest("BUS-1", *north(1100), ts=t); t += 5
        t = drive(tr, 1150, 2200, t)
        v = tr.vehicles["BUS-1"]
        self.assertEqual(got, [], "a normal trip must not raise alerts")
        self.assertEqual([s["status"] for s in v.stops], ["done", "done", "done"])
        self.assertEqual(v.trip, "completed")
        self.assertEqual(tr.snapshot()["vehicles"][0]["status"], "COMPLETED")

    def test_stop_status_goes_pending_arrived_done(self):
        tr = make()
        v = tr.vehicles["BUS-1"]
        drive(tr, 0, 450, 1000)
        self.assertEqual(v.stops[0]["status"], "pending")
        tr.ingest("BUS-1", *north(540), ts=2000)
        self.assertEqual(v.stops[0]["status"], "arrived")
        tr.ingest("BUS-1", *north(700), ts=2010)
        self.assertEqual(v.stops[0]["status"], "done")


class RouteDeviation(unittest.TestCase):
    def test_deviation_raised_after_three_readings_then_cleared(self):
        tr = make()
        t = drive(tr, 0, 300, 1000)
        t = drive(tr, 350, 450, t, offset_east=200)               # 3 readings, 200 m off the route
        self.assertIn("ROUTE_DEVIATION", kinds(tr))
        self.assertEqual(tr.snapshot()["vehicles"][0]["status"], "OFF ROUTE")
        drive(tr, 500, 650, t)                                    # back on the route
        self.assertNotIn("ROUTE_DEVIATION", kinds(tr))

    def test_one_or_two_noisy_readings_are_ignored(self):
        tr = make()
        t = drive(tr, 0, 300, 1000)
        t = drive(tr, 350, 400, t, offset_east=200)               # only 2 readings
        drive(tr, 450, 600, t)
        self.assertEqual(kinds(tr), [])
        self.assertEqual(tr.db.execute("SELECT COUNT(*) FROM transport_alert").fetchone()[0], 0)

    def test_a_400m_teleport_is_a_gps_glitch_not_a_deviation(self):
        tr = make()
        t = drive(tr, 0, 300, 1000)
        drive(tr, 350, 450, t, offset_east=400)                   # impossible 80 m/s sideways jump
        self.assertEqual(kinds(tr), [])

    def test_small_offsets_are_fine(self):
        tr = make()
        drive(tr, 0, 1000, 1000, offset_east=60)
        self.assertEqual(kinds(tr), [])


class UnexpectedStop(unittest.TestCase):
    def test_long_stop_away_from_stops_raises_alert_then_clears_when_moving(self):
        tr = make()
        t = drive(tr, 0, 300, 1000)
        p = north(320)                                            # 230 m from Stop A
        for k in range(20):                                       # 200 s standing still, with GPS noise
            tr.ingest("BUS-1", p[0] + (k % 3) * 4 * M, p[1], ts=t); t += 10
        self.assertIn("UNEXPECTED_STOP", kinds(tr))
        self.assertEqual(tr.snapshot()["vehicles"][0]["status"], "STOPPED")
        drive(tr, 400, 450, t)
        self.assertNotIn("UNEXPECTED_STOP", kinds(tr))

    def test_short_stop_is_fine(self):
        tr = make()
        t = drive(tr, 0, 300, 1000)
        for k in range(8):                                        # 80 s
            tr.ingest("BUS-1", *north(320), ts=t); t += 10
        self.assertEqual(kinds(tr), [])

    def test_waiting_at_a_planned_stop_is_not_an_alert(self):
        tr = make()
        t = drive(tr, 0, 500, 1000)
        for k in range(40):                                       # 400 s at Stop A
            tr.ingest("BUS-1", *north(550), ts=t); t += 10
        self.assertEqual(kinds(tr), [])

    def test_waiting_at_the_start_is_not_an_alert(self):
        tr = make()
        for k in range(40):
            tr.ingest("BUS-1", *north(5), ts=1000 + k * 10)
        self.assertEqual(kinds(tr), [])

    def test_no_alerts_after_trip_completed(self):
        tr = make()
        t = drive(tr, 0, 2200, 1000)
        self.assertEqual(tr.vehicles["BUS-1"].trip, "completed")
        for k in range(60):                                       # parked at the office for 10 min
            tr.ingest("BUS-1", *north(2205), ts=t); t += 10
        self.assertEqual(kinds(tr), [])


class SignalLost(unittest.TestCase):
    def test_signal_lost_and_recovered(self):
        tr = make()
        tr.ingest("BUS-1", *north(100), ts=1000, rx=5000)
        tr.check_signal(now=5030)
        self.assertEqual(kinds(tr), [])                           # 30 s is fine
        tr.check_signal(now=5100)
        self.assertEqual(kinds(tr), ["SIGNAL_LOST"])
        self.assertEqual(tr.snapshot(now=5100)["vehicles"][0]["status"], "NO SIGNAL")
        tr.ingest("BUS-1", *north(150), ts=1010, rx=5110)
        self.assertEqual(kinds(tr), [])

    def test_no_signal_alert_for_vehicle_that_never_started(self):
        tr = make()
        tr.check_signal(now=99999)
        self.assertEqual(kinds(tr), [])


class StopsMissed(unittest.TestCase):
    def test_skipping_a_stop_is_reported(self):
        got = []
        tr = make(notify=got.append)
        tr.ingest("BUS-1", *north(10), ts=1000)
        tr.ingest("BUS-1", *north(1100), ts=1100)                 # arrives at Stop B, never visited Stop A
        v = tr.vehicles["BUS-1"]
        self.assertEqual([s["status"] for s in v.stops], ["missed", "arrived", "pending"])
        self.assertEqual([a["type"] for a in got], ["STOP_MISSED"])


class BadData(unittest.TestCase):
    def test_gps_jump_is_ignored(self):
        tr = make()
        tr.ingest("BUS-1", *north(100), ts=1000)
        r = tr.ingest("BUS-1", north(100)[0] + 0.5, LON, ts=1005)   # 55 km in 5 s
        self.assertFalse(r["accepted"])
        self.assertAlmostEqual(tr.vehicles["BUS-1"].last["lat"], north(100)[0], places=6)

    def test_two_readings_a_split_second_apart_are_not_a_glitch(self):
        tr = make()
        tr.ingest("BUS-1", *north(100), ts=1000.00)
        self.assertTrue(tr.ingest("BUS-1", *north(112), ts=1000.05)["accepted"])   # 12 m of GPS noise

    def test_old_and_duplicate_readings_are_ignored(self):
        tr = make()
        tr.ingest("BUS-1", *north(100), ts=1000)
        self.assertFalse(tr.ingest("BUS-1", *north(110), ts=1000)["accepted"])
        self.assertFalse(tr.ingest("BUS-1", *north(110), ts=900)["accepted"])

    def test_invalid_input_is_rejected(self):
        tr = make()
        with self.assertRaises(KeyError):
            tr.ingest("NOPE", 12.9, 77.6)
        with self.assertRaises(ValueError):
            tr.ingest("BUS-1", 123.0, 77.6)
        with self.assertRaises(ValueError):
            tr.ingest("BUS-1", float("nan"), 77.6)
        with self.assertRaises(ValueError):
            tr.ingest("BUS-1", "abc", 77.6)


class Housekeeping(unittest.TestCase):
    def test_everything_is_saved_in_the_database(self):
        tr = make()
        drive(tr, 0, 450, 1000, offset_east=0)
        drive(tr, 500, 700, 2000, offset_east=200)
        n_log = tr.db.execute("SELECT COUNT(*) FROM gps_log").fetchone()[0]
        n_alert = tr.db.execute("SELECT COUNT(*) FROM transport_alert WHERE alert_type='ROUTE_DEVIATION'").fetchone()[0]
        self.assertEqual(n_log, 15)       # 10 on route + 5 off route
        self.assertEqual(n_alert, 1)
        self.assertEqual(tr.db.execute("SELECT vehicle_no FROM vehicle").fetchone()[0], "Bus 1")

    def test_reset_clears_the_trip(self):
        tr = make()
        drive(tr, 0, 2200, 1000)
        tr.reset("BUS-1")
        v = tr.vehicles["BUS-1"]
        self.assertEqual(v.trip, "not_started")
        self.assertEqual([s["status"] for s in v.stops], ["pending"] * 3)

    def test_snapshot_is_json_friendly(self):
        import json
        tr = make()
        drive(tr, 0, 300, 1000)
        json.dumps(tr.snapshot())


if __name__ == "__main__":
    unittest.main(verbosity=2)
