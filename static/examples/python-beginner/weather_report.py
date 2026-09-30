"""Create a seven-day weather report; use --live to fetch Open-Meteo data."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save charts on machines without a desktop display.
import matplotlib.pyplot as plt
import pandas as pd
import requests

SAMPLE = {
    "daily": {
        "time": [f"2026-01-{day:02d}" for day in range(1, 8)],
        "temperature_2m_min": [3, 4, 2, 1, 3, 5, 4],
        "temperature_2m_max": [9, 10, 8, 7, 10, 12, 11],
    }
}


def fetch_weather(latitude, longitude):
    """Fetch the seven completed days before today in the location's timezone."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min",
            "past_days": 7,
            "forecast_days": 0,
            "timezone": "auto",
            "temperature_unit": "celsius",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def weather_table(payload):
    """Convert the API's parallel arrays into a checked table."""
    daily = payload["daily"]
    table = pd.DataFrame(
        {
            "date": daily["time"],
            "min_temp": daily["temperature_2m_min"],
            "max_temp": daily["temperature_2m_max"],
        }
    )
    if len(table) != 7:
        raise ValueError("Expected exactly seven daily rows")
    table["date"] = pd.to_datetime(table["date"], errors="raise")
    for column in ("min_temp", "max_temp"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table.isna().any().any():
        raise ValueError("Weather data contains missing values")
    if not table["date"].is_unique:
        raise ValueError("Weather data contains duplicate dates")
    table = table.sort_values("date").reset_index(drop=True)
    gaps = table["date"].diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError("Weather dates must be consecutive")
    if not (table["min_temp"] <= table["max_temp"]).all():
        raise ValueError("Minimum temperature exceeds maximum")
    return table


def save_report(table, output_dir, title):
    """Save both reusable data and a labelled chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "weather.csv", index=False, date_format="%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(table["date"], table["min_temp"], marker="o", label="Minimum")
    ax.plot(table["date"], table["max_temp"], marker="o", label="Maximum")
    ax.set(title=title, xlabel="Date", ylabel="Temperature (°C)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "weather.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--latitude", type=float, default=48.8566)
    parser.add_argument("--longitude", type=float, default=2.3522)
    parser.add_argument("--city", default="Paris")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "weather",
    )
    args = parser.parse_args()
    if not -90 <= args.latitude <= 90 or not -180 <= args.longitude <= 180:
        parser.error("Latitude must be -90..90 and longitude -180..180")
    try:
        payload = fetch_weather(args.latitude, args.longitude) if args.live else SAMPLE
        table = weather_table(payload)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Unable to prepare weather data: {exc}\n")
    title = (
        f"{args.city}: previous seven days (Open-Meteo)"
        if args.live
        else "Practice weather: synthetic sample"
    )
    save_report(table, args.output_dir, title)
    print(table.to_string(index=False))
    print(f"Saved {len(table)} rows and chart to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
