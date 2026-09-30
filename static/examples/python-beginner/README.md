# Python beginner labs

Original practice implementations accompanying the Python lessons, informed by
[Dave Ebbelaar's Python for AI course](https://www.youtube.com/watch?v=ygXn5nV5qFc).
The included weather sample is synthetic; the sales CSV is practice data.

Use Python 3.12 or later. From this directory on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python weather_report.py
python sales_analysis/analyzer.py
```

On Windows, create the environment with `py -m venv .venv`, then activate with
`.\.venv\Scripts\Activate.ps1`. The remaining `python` commands are the same.

The weather script defaults to an offline sample and writes `output/weather/weather.csv`
and `weather.png`. Fetch seven completed days from Open-Meteo with:

```bash
python weather_report.py --live --city Paris --latitude 48.8566 --longitude 2.3522
```

Live mode requires internet access and the public API to be available. Data is from
[Open-Meteo](https://open-meteo.com/), provided under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain attribution when
sharing live reports. The default fixture is not an observed Paris weather record.

The sales script writes three formats under `sales_analysis/output`. Expected row
totals are 120, 50 and 50; the grand total is 220. These are simple educational
calculations, not an accounting system. For money requiring exact decimal rules,
use integer minor units or `decimal.Decimal` and define rounding explicitly.

Both scripts accept `--output-dir PATH` and replace their named output files on
each run. The sales script also accepts `--input PATH`. Imports do not run reports.
Paths supplied as command-line arguments are relative to your working directory;
default paths are anchored to the scripts.

The requirements file declares compatible direct dependencies rather than a tested
lockfile. To manage a separate project with uv, use `uv init`, add these libraries
with `uv add`, then commit the generated `uv.lock` alongside `pyproject.toml`.
