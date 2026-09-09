"""Serial NMEA GNSS/RTK reader.

Ported from RTK-Test (pynmea2 + pyserial). Parses GGA sentences for the
position/fix and RMC for the UTC date, then emits a GPSFix per valid fix.
Run standalone (`python rtk_gps.py --port ...`) to behave like the original
RTK-Test CLI, or import GNSSReader to stream fixes via a callback.
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pynmea2
import serial
from serial import SerialException

FIX_QUALITY = {
    0: "No Fix",
    1: "GPS (SPS)",
    2: "DGPS",
    3: "PPS",
    4: "RTK Fixed",
    5: "RTK Float",
    6: "Dead Reck.",
    7: "Manual",
    8: "Simulation",
}

# Horizontal / vertical accuracy (m) baseline per fix quality. RTK gives
# cm-level, float decimetric, single-point a few meters.
_BASE_ACCURACY = {
    0: (100.0, 150.0),
    1: (2.5, 4.0),
    2: (0.8, 1.5),
    4: (0.02, 0.04),
    5: (0.2, 0.4),
}


@dataclass
class GPSFix:
    lat: float
    lon: float
    alt_msl: float
    fix_quality: int
    num_sats: int
    hdop: float
    unix_time: float
    latitude_std: float
    longitude_std: float
    altitude_std: float

    @property
    def fix_name(self) -> str:
        return FIX_QUALITY.get(self.fix_quality, f"Unknown ({self.fix_quality})")


def estimate_accuracy(fix_quality: int, hdop: float) -> tuple[float, float, float]:
    """Return (latitude_std, longitude_std, altitude_std) in meters."""
    h_base, v_base = _BASE_ACCURACY.get(fix_quality, _BASE_ACCURACY[0])
    factor = 1.0
    if hdop and 0.5 <= hdop <= 20.0:
        factor = hdop
    h_std = h_base * factor
    v_std = v_base * max(1.0, factor)
    return h_std, h_std, v_std


class GNSSReader:
    """Reads NMEA from a serial GNSS/RTK receiver in a background thread."""

    def __init__(self, port: str, baud: int = 115200, retry_delay: float = 3.0,
                 on_fix=None) -> None:
        self.port = port
        self.baud = baud
        self.retry_delay = retry_delay
        self.on_fix = on_fix
        self._stop = threading.Event()
        self._thread = None
        self._utc_date = date.today()
        self.last_fix: GPSFix | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _connect(self) -> serial.Serial:
        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.port, self.baud, timeout=0.5)
                print(f"[gnss] Connected to {self.port} @ {self.baud} baud", flush=True)
                return ser
            except SerialException as e:
                # Hint for common Jetson pitfalls: dialout group or ModemManager
                hint = ""
                if "Permission denied" in str(e):
                    hint = " (hint: user not in 'dialout' group or needs re-login; try 'sudo usermod -aG dialout $USER' then logout, or 'newgrp dialout'; also check ModemManager: 'sudo systemctl stop ModemManager' or add udev rule ENV{ID_MM_DEVICE_IGNORE}=\"1\")"
                print(f"[gnss] Port {self.port} unavailable ({e}){hint}. "
                      f"Retrying in {self.retry_delay}s...", flush=True)
                self._stop.wait(self.retry_delay)
        raise SerialException("stopped")

    def _run(self) -> None:
        ser = None
        while not self._stop.is_set():
            try:
                if ser is None:
                    ser = self._connect()
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = pynmea2.parse(line)
                except pynmea2.ParseError:
                    continue

                if isinstance(msg, pynmea2.types.RMC) and msg.datestamp:
                    self._utc_date = msg.datestamp
                    continue
                if not isinstance(msg, pynmea2.types.GGA):
                    continue
                if not msg.latitude or not msg.longitude or msg.gps_qual == 0:
                    continue
                if msg.timestamp is None:
                    continue

                fix = self._make_fix(msg)
                self.last_fix = fix
                if self.on_fix is not None:
                    try:
                        self.on_fix(fix)
                    except Exception as e:
                        print(f"[gnss] on_fix error: {e}", flush=True)
            except SerialException:
                print("\n[gnss] Serial connection lost. Reconnecting...", flush=True)
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
                ser = None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"[gnss] read error: {e}", flush=True)
                time.sleep(0.5)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _make_fix(self, msg) -> GPSFix:
        lat = msg.latitude if msg.lat_dir in ("N", "") else -msg.latitude
        lon = msg.longitude if msg.lon_dir in ("E", "") else -msg.longitude
        alt = float(msg.altitude) if msg.altitude else 0.0
        hdop = float(msg.horizontal_dil) if msg.horizontal_dil else 0.0
        dt = datetime.combine(self._utc_date, msg.timestamp, tzinfo=timezone.utc)
        lat_std, lon_std, alt_std = estimate_accuracy(msg.gps_qual, hdop)
        return GPSFix(
            lat=lat,
            lon=lon,
            alt_msl=alt,
            fix_quality=msg.gps_qual,
            num_sats=int(msg.num_sats),
            hdop=hdop,
            unix_time=dt.timestamp(),
            latitude_std=lat_std,
            longitude_std=lon_std,
            altitude_std=alt_std,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTK GPS reader")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--retry-delay", type=float, default=3.0,
                        help="Seconds between reconnect attempts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def on_fix(fix: GPSFix) -> None:
        print(
            f"\r[{fix.lat:.6f}, {fix.lon:.6f}]  "
            f"Alt: {fix.alt_msl:.1f} m  "
            f"Fix: {fix.fix_name}  "
            f"Sats: {fix.num_sats}  "
            f"(Ctrl+C to quit)   ",
            end="",
            flush=True,
        )

    reader = GNSSReader(args.port, args.baud, args.retry_delay, on_fix=on_fix)
    reader.start()
    print("Waiting for GPS fix... (Ctrl+C to quit)", flush=True)
    try:
        while True:
            if not reader._thread.is_alive():
                print("\nReader stopped.", flush=True)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting.", flush=True)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
