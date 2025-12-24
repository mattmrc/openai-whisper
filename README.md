# MLX Whisper CLI

A minimal CLI wrapper around `mlx-whisper` for transcribing audio on Apple Silicon.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a CLI:

```bash
pip install -e .
whisper-cli --help
```

## Development

```bash
pip install -e .[dev] --no-deps
pytest
```

## Usage

```bash
python run_whisper.py /path/to/audio.m4a
```

Common options:

```bash
python run_whisper.py audio.m4a \
  --model mlx-community/whisper-large-v3-turbo \
  --output-format srt \
  --language en
```

Formats: `txt`, `srt`, `vtt`, `json`.

Outputs default to the `transcripts/` directory (created automatically).

Batch mode:

```bash
python run_whisper.py audio1.m4a audio2.m4a --output-format json
```

Logging and output:

```bash
python run_whisper.py audio.m4a --verbose
python run_whisper.py audio.m4a --stdout
python run_whisper.py audio.m4a --metadata
```

Performance and cache:

```bash
python run_whisper.py audio.m4a --chunk-length 30 --stride-length 5 --batch-size 8
python run_whisper.py audio.m4a --cache-dir ~/.cache/whisper
```

Decoding options:

```bash
python run_whisper.py audio.m4a --beam-size 5 --temperature 0.0
python run_whisper.py audio.m4a --logprob-threshold -1.0 --no-speech-threshold 0.6
```

## Config

Create a `.whisper.toml` in the repo root (or pass `--config path/to/config.toml`):

```toml
model = "mlx-community/whisper-large-v3-turbo"
output_format = "srt"
language = "en"
verbose = 1
```

## Notes

- Models are downloaded from Hugging Face on first use.
- MLX Whisper is intended for Apple Silicon machines.

## Exit codes

- `0`: success
- `1`: runtime error (transcription or write failure)
- `2`: usage/config error

## Docs

- `docs/guide.md`
