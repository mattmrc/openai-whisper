# Guide

## Model selection

- `mlx-community/whisper-base-mlx`: fastest, lowest accuracy.
- `mlx-community/whisper-large-v3-turbo`: best balance for meetings.
- `mlx-community/whisper-large-v3-mlx`: highest accuracy, slower.

## Performance tips

- Use `--chunk-length` and `--stride-length` for long audio to reduce memory use.
- Increase `--batch-size` for throughput if you have memory headroom.
- Use `--beam-size` for higher accuracy at the cost of speed.

## Output formats

- `txt`: plain transcript.
- `srt`: subtitle format with comma timestamps.
- `vtt`: WebVTT subtitles with dot timestamps.
- `json`: full MLX Whisper output plus `meta` field.

## Output directory

Transcripts are written to `transcripts/` by default and the folder is created
automatically.

## Troubleshooting

- If model downloads are slow, set `--cache-dir` to a fast local disk.
- For unexpected silence, lower `--no-speech-threshold`.
