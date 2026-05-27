# hocloop-proxy-model

A command-line tool for running the Hocloop proxy model pipeline.

## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation
```bash
uv sync
```

## Usage

``bash
uv run main.py --features-file "path" --targets-file "path"  --output-path "path"
``

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--features-file` | ✅ | Path to the features file (with header) |
| `--targets-file` | ✅ | Path to the targets file (with header) |
| `--output-path` | ✅ | Path to the output file |

### Example
```bash
uv run main.py
--features-file data/features.csv
--targets-file data/targets.csv
--output-path results
```


## Logging

Logs are written to both the console and `hocloop-proxy-model.log` in the working directory.