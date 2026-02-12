import json
import os
from datetime import datetime, timedelta, timezone

import pandas_market_calendars as mcal
import subprocess

CACHE = ".github/last_run.json"

def load():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}

def save(x):
    json.dump(x, open(CACHE, "w"))

def main():
    now = datetime.now(timezone.utc)

    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(
        start_date=(now.date() - timedelta(days=3)).isoformat(),
        end_date=(now.date() + timedelta(days=3)).isoformat(),
    )

    past = sched[sched["market_close"] <= now]
    if past.empty:
        print("No close yet")
        return

    last_close = past["market_close"].iloc[-1].to_pydatetime()
    run_after = last_close + timedelta(minutes=15)

    if now < run_after:
        print("Too early")
        return

    day = last_close.date().isoformat()

    last = load()
    if last.get("day") == day:
        print("Already ran today")
        return

    print("Running scan...")
    subprocess.check_call(["python", "scan.py"])

    save({"day": day})
    print("Done")

if __name__ == "__main__":
    main()
