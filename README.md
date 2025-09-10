# IFPILM
Code used to analyze data collected by PlaNS thrust stand

# Overview

This repository contains three independent analysis/collection scripts:
- pogodynka.py — weather collector (scrapes a station page and Windy embed, writes a CSV log).
- daily.py — daily / multi-day analysis and plotting (merges measurement time series, computes noise/derivatives, correlates with weather).
- impulse_data.py — impulse / engine-run analysis (detects PPT vs Hall engine signals and analyzes each using ppt and hall modules).

Each script is intended to be runnable standalone.

# Requirements

Minimum / typical Python dependencies (install with pip):

`python 3.8+`

`pip install numpy scipy pandas matplotlib requests pillow logzero selenium`

### Additional notes:

pogodynka.py uses Selenium + a Chrome browser. You must have Chrome (or Chromium) and a matching ChromeDriver installed and available in PATH. The code uses Service() with no explicit path, so ChromeDriver needs to be discoverable by Selenium. (The code contains a commented-out webdriver_manager import — you may enable that workflow if you prefer automatic driver management.)

hall and ppt are local Python modules (present in the repo). They depend on scipy and numpy.

# Description
## pogodynka.py
Continuously collects weather data from:
- a local station page on traxelektronik.pl (fetched via requests), and
- Windy (embedded weather widget) using Selenium to extract the selected-hour wind temperature/speed/gusts.

It creates a CSV file and appends a new row every loop iteration (the script loops forever with a 10-minute wait between successful writes).

### Important behaviour details you should know

The script sleeps 30s after loading Windy, then extracts specific XPaths for hour/temperature/wind. If the Windy markup changes, the XPath may break.

The code assumes the station response.text contains specific substrings — the parsing is brittle and text-splitting based.

The created CSV format (semicolon-separated) is intentionally matched by daily.py (see below).

## daily.py
Analyze a single day or multiple days of recorded displacement data, compute noise & derivatives, merge with PT100 temperature logs, and compute correlations between displacement (or its derivative) and various weather variables.

This script creates time-series plots (and noise/correlation plots) saved under plots/.
> [!IMPORTANT]
> Input paths required (as used in the code):

> Measurement/time-series file(s):
`/dane dobowe/<YYYYMMDD>/TB/00`
(each foldername is expected to be YYYYMMDD like 20250901.)

> Per-day PT100 log:
`/dane dobowe/<YYYYMMDD>/picolog.csv`

> Weather CSV that pogodynka.py helps to create:
`Pogoda {dates}.csv`

where dates is the dates string used inside multiple() (e.g. "01-08").

PNG plots saved under plots/ (filenames include the dates token and the plot subtitle). 
> [!NOTE]
> Derivative plots if `derivative=True`.

Console output with cross-correlation results (max/min and lag/time) for each tested weather parameter.

## impulse_data.py
Analyze experimental impulse / engine test data stored under a configurable work_folder. The script decides whether a measurement file is a Hall engine (long record) or a PPT engine (short record) and runs the appropriate analysis routine (hall.analyze_hall() for Hall; ppt.analyze_ppt() for PPT). It can also compute a calibration-derived mass/impulse using a separate “calibrator” folder.
> [!IMPORTANT]
> Inputs required

> `work_folder` — folder containing data files. Each file is expected to be tab-separated and follow the Time [ms] header format above.

Outputs produced by impulse_data.py

- Console output describing per-file analysis (prints “I think it's a Hall engine!” or “I think it's a PPT engine!”, thrust/impulse values, and error statistics).

- Interactive plots shown via matplotlib.pyplot.show() if enable_plot = True. (Important) the script does not save plots to disk by default (unlike daily.py).
