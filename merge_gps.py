"""
merge_tr_line_times_unix.py — объединяет GNSS TR-данные с логом line_times
и создаёт CSV с Unix-временем (секунды с 1970-01-01 UTC) и микросекундами.

Сетка: 10 Гц (0.1 с), только пересечение диапазонов TR и line_times.
"""

import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from bisect import bisect_right
import numpy as np
import pandas as pd

MSK_OFFSET = timedelta(hours=3)
LEAP_SECOND = timedelta(seconds=18)

def log(msg):
    print(f"[LOG] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")

# === Парсинг TR ===
def parse_tr_file(path: Path) -> pd.DataFrame:
    started, rows = False, []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not started:
                if "GPSTime" in line and "Latitude" in line and "Longitude" in line:
                    started = True
                continue
            if re.search(r"\(sec\).*?\(deg\).*?\(deg\).*?\(m\)", line):
                continue
            p = line.strip().split()
            if len(p) < 5:
                continue
            try:
                gpstime = float(p[1])
                lat = float(p[2])
                lon = float(p[3])
                h = float(p[4])
            except Exception:
                continue
            rows.append((gpstime, lat, lon, h))
    if not rows:
        raise RuntimeError("Не удалось найти табличные данные в TR.")
    return pd.DataFrame(rows, columns=["gpstime_s", "Latitude_deg", "Longitude_deg", "H_Ell_m"])

# === Парсинг line_times ===
def parse_line_times(path: Path) -> pd.DataFrame:
    started_dt, times, idxs = None, [], []
    rex = re.compile(r"^Line\s+(\d+)\s+(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)$")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if s.startswith("Started:"):
                tstr = s.split("Started:")[1].strip()
                try:
                    started_dt = datetime.fromisoformat(tstr)
                except Exception:
                    started_dt = datetime.strptime(tstr, "%Y-%m-%dT%H:%M:%S")
                continue
            m = rex.match(s)
            if not m:
                continue
            if started_dt is None:
                raise RuntimeError("Нет 'Started:' перед Line ...")
            idx = int(m.group(1))
            tod = m.group(2)
            fmt = "%H:%M:%S.%f" if "." in tod else "%H:%M:%S"
            t = datetime.strptime(tod, fmt)
            abs_t = started_dt.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond)
            if abs_t < started_dt:
                abs_t += timedelta(days=1)
            idxs.append(idx)
            times.append(abs_t.timestamp())
    return pd.DataFrame({"line_index": idxs, "time_s": times})

# === Основная функция ===
def resample_tr_to_unix(tr_df, lines_df, first_gpstime, tr_start_local, rate_hz=10.0):
    """Интерполяция TR в Unix-время (10 Гц, только пересечение диапазонов)."""
    # Определяем начало GPS-недели
    week_start_utc = tr_start_local - MSK_OFFSET
    week_start_utc -= timedelta(days=(week_start_utc.weekday() + 1) % 7)
    week_start_utc = week_start_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    gpstime = tr_df["gpstime_s"].to_numpy()
    # абсолютное UTC-время TR
    t_tr_utc = week_start_utc + np.array([timedelta(seconds=t) for t in gpstime])
    # перевод в системное время (MSK, с вычетом 18 с)
    t_tr_sync = np.array([t + MSK_OFFSET - LEAP_SECOND for t in t_tr_utc])
    t_tr_unix = np.array([t.timestamp() for t in t_tr_sync])

    # времена линий
    t_lines = np.sort(lines_df["time_s"].to_numpy())

    # пересечение диапазонов
    start_abs = max(t_tr_unix[0], t_lines[0])
    end_abs = min(t_tr_unix[-1], t_lines[-1])
    if start_abs >= end_abs:
        print("⚠️ Нет пересечения по времени.")
        return pd.DataFrame()

    print(f"[INFO] Пересечение времён: {datetime.fromtimestamp(start_abs)} .. {datetime.fromtimestamp(end_abs)}")

    # 10 Гц сетка в Unix-времени
    dt = 1.0 / rate_hz
    n_points = int((end_abs - start_abs) / dt) + 1
    t_grid_unix = start_abs + np.arange(n_points) * dt

    # Интерполяция координат
    lat = np.interp(t_grid_unix, t_tr_unix, tr_df["Latitude_deg"])
    lon = np.interp(t_grid_unix, t_tr_unix, tr_df["Longitude_deg"])
    h = np.interp(t_grid_unix, t_tr_unix, tr_df["H_Ell_m"])

    # Подсчёт линий
    counts = [bisect_right(t_lines, t) for t in t_grid_unix]

    # Разделение на UnixTime и Microseconds
    unix_int = np.floor(t_grid_unix).astype(np.int64)
    micro = ((t_grid_unix - unix_int) * 1e6).astype(np.int64)

    df = pd.DataFrame({
        "Lines": counts,
        "UnixTime": unix_int,
        "Microseconds": micro,
        "Latitude": lat,
        "Longitude": lon,
        "Altitude": h,
    })

    # Заполняем оставшиеся поля нулями
    for c in [
        "W", "X", "Y", "Z",
        "XVelocity", "YVelocity", "ZVelocity",
        "UAVTimeMs", "SysUnixTime", "SysMicroseconds"
    ]:
        df[c] = 0

    return df

# === Сохранение ===
def save_csv(df, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(df.columns) + "\n")
        for r in df.itertuples(index=False):
            f.write(",".join(str(x) for x in r) + "\n")

def auto_extract_from_tr(path: Path):
    """
    Достаёт:
      - время обработки из строки ProcessInfo
      - первое GPSTime из первой строки таблицы TR
    Возвращает (first_gpstime, tr_start_local).
    """

    log(f"Читаю TR-файл: {path}")

    process_dt = None
    first_gpstime = None

    rex_proc = re.compile(
        r"ProcessInfo:.*on\s+(\d{1,2}/\d{1,2}/\d{4})\s+at\s+(\d{2}:\d{2}:\d{2})"
    )
    rex_row = re.compile(r"^\S+\s+([0-9]+\.[0-9]+)\s")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if process_dt is None:
                m = rex_proc.search(line)
                if m:
                    date_str, time_str = m.group(1), m.group(2)
                    process_dt = datetime.strptime(
                        date_str + " " + time_str, "%m/%d/%Y %H:%M:%S"
                    )
                    log(f"Нашёл время обработки TR: {process_dt}")

            if first_gpstime is None:
                m2 = rex_row.match(line.strip())
                if m2:
                    try:
                        first_gpstime = float(m2.group(1))
                        log(f"Нашёл первое GPSTime: {first_gpstime}")
                    except Exception:
                        pass

            if process_dt and first_gpstime:
                break

    if process_dt is None:
        err("Не найден ProcessInfo — TR-файл не standard IE?")
        raise RuntimeError("ProcessInfo не найден")

    if first_gpstime is None:
        err("Не найдено первое GPSTime — таблица TR повреждена?")
        raise RuntimeError("GPSTime не найден")

    if first_gpstime < 0 or first_gpstime > 700000:
        warn(f"Похоже странный GPSTime ({first_gpstime}) — неделя имеет 604800 секунд!")

    # Вычисление времени начала TR
    week_start_utc = process_dt - MSK_OFFSET
    week_start_utc -= timedelta(days=(week_start_utc.weekday() + 1) % 7)
    week_start_utc = week_start_utc.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    tr_start_utc = week_start_utc + timedelta(seconds=first_gpstime)
    tr_start_local = tr_start_utc + MSK_OFFSET - LEAP_SECOND

    log(f"Вычислил локальное время первой строки TR: {tr_start_local}")

    return first_gpstime, tr_start_local

# === CLI ===
def main(argv=None):
    ap = argparse.ArgumentParser(description="TR + line_times → Unix CSV (10 Гц, пересечение).")
    ap.add_argument("--tr", required=True, type=Path)
    ap.add_argument("--lines", required=True, type=Path)
    ap.add_argument("-o", "--out", default=Path("merged_unix.csv"), type=Path)
    args = ap.parse_args(argv)

    # --- Автоматическое извлечение параметров ---
    try:
        first_gpstime, tr_start_local = auto_extract_from_tr(args.tr)
    except Exception as e:
        err(str(e))
        return 1

    # --- Парсинг TR ---
    log("Парсинг TR...")
    tr_df = parse_tr_file(args.tr)
    log(f"TR содержит {len(tr_df)} строк данных.")

    # --- Парсинг line_times ---
    log("Парсинг line_times...")
    lines_df = parse_line_times(args.lines)
    log(f"line_times содержит {len(lines_df)} строк.")

    if len(lines_df) == 0:
        err("line_times пуст — нечего сопоставлять.")
        return 1

    # --- Пересчёт ---
    log("Выполняю интерполяцию и совмещение...")
    df = resample_tr_to_unix(tr_df, lines_df, first_gpstime, tr_start_local, 10.0)

    if df.empty:
        warn("Нет пересечения данных по времени — файл не будет сохранён.")
        return 0

    # --- Сохранение ---
    save_csv(df, args.out)
    log(f"OK: сохранено {len(df)} строк → {args.out}")
    log(f"Unix диапазон: {df['UnixTime'].iloc[0]} ... {df['UnixTime'].iloc[-1]}")

    # Контроль структуры
    required_cols = ["UnixTime", "Microseconds", "Latitude", "Longitude"]
    for col in required_cols:
        if col not in df.columns:
            warn(f"Проблема: отсутствует столбец {col}")

    log("Готово.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())