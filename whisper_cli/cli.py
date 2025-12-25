import argparse
import inspect
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    tomllib = None

try:
    import tomli  # type: ignore[import-not-found]
except ModuleNotFoundError:
    tomli = None

try:
    import mlx_whisper
except ModuleNotFoundError:
    mlx_whisper = None

# Options:
# "mlx-community/whisper-base-mlx"
# "mlx-community/whisper-large-v3-turbo"
# "mlx-community/whisper-large-v3-mlx"
MODEL_NAME = "mlx-community/whisper-large-v3-turbo"

SUPPORTED_FORMATS = ("txt", "srt", "vtt", "json")

DEFAULT_CONFIG_NAME = ".whisper.toml"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_RUNTIME = 1

LOGGER = logging.getLogger("whisper")


def format_elapsed(seconds: float) -> str:
    """Return a human-readable duration.

    - < 60 seconds: "X.XX seconds"
    - >= 60 seconds: "M min SS sec"
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = int(seconds // 60)
    remaining = int(round(seconds - minutes * 60))
    return f"{minutes} min {remaining:02d} sec"


def find_config_path(argv: Iterable[str]) -> Path | None:
    argv_list = list(argv)
    if "--config" in argv_list:
        idx = argv_list.index("--config")
        if idx + 1 < len(argv_list):
            return Path(argv_list[idx + 1]).expanduser()
    return None


def load_config(config_path: Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() != ".toml":
        raise ValueError("Config file must be TOML.")

    data = config_path.read_bytes()
    parser = tomllib if tomllib is not None else tomli
    if parser is None:
        raise RuntimeError("TOML config requested but tomllib/tomli not available.")
    config = parser.loads(data.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Invalid config format; expected a TOML table.")
    return config


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    for key in ("output_dir", "cache_dir", "config"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = Path(value)
    return normalized


def parse_args(argv: Iterable[str], config: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with MLX Whisper.",
        epilog=(
            "Examples:\n"
            "  whisper-cli audio.m4a --output-format srt\n"
            "  whisper-cli audio.m4a --language en --beam-size 5\n"
            "  whisper-cli audio.m4a --temperature 0.0 --no-speech-threshold 0.6\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="Path(s) to audio files to transcribe.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML config file.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Hugging Face repo or local path for the MLX Whisper model.",
    )
    parser.add_argument(
        "--output-format",
        choices=SUPPORTED_FORMATS,
        default="txt",
        help="Output format for the transcript.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory to write transcripts into.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript files.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write transcript to stdout (single file only).",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Write a sidecar metadata JSON file alongside the transcript.",
    )
    parser.add_argument(
        "--language",
        help="Force a language code (e.g., en, es).",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        help="Transcription task type.",
    )
    parser.add_argument(
        "--chunk-length",
        type=float,
        help="Chunk length in seconds for long audio.",
    )
    parser.add_argument(
        "--stride-length",
        type=float,
        help="Stride/overlap in seconds between chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for decoding.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="Beam size for decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature for decoding.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        help="Compression ratio threshold for detecting hallucinations.",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        help="Log probability threshold for low-confidence segments.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        help="No-speech probability threshold for skipping segments.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word timestamps in output segments when supported.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Override cache directory for model downloads.",
    )
    parser.add_argument(
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use twice for debug).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error logs.",
    )

    if config:
        parser.set_defaults(**config)

    return parser.parse_args(list(argv))


def configure_logging(verbose: int, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def format_timestamp(seconds: float, use_comma: bool) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if use_comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def build_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(float(segment["start"]), use_comma=True)
        end = format_timestamp(float(segment["end"]), use_comma=True)
        text = str(segment.get("text", "")).strip()
        lines.extend([str(index), f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip() + "\n"


def build_vtt(segments: list[dict]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for segment in segments:
        start = format_timestamp(float(segment["start"]), use_comma=False)
        end = format_timestamp(float(segment["end"]), use_comma=False)
        text = str(segment.get("text", "")).strip()
        lines.extend([f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip() + "\n"


def write_output(
    *,
    output_path: Path,
    output_format: str,
    result: dict,
    overwrite: bool,
    metadata: dict[str, Any],
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    if output_format == "txt":
        text = str(result.get("text", "")).strip()
        output_path.write_text(text + "\n", encoding="utf-8")
        return

    if output_format == "json":
        payload = result if isinstance(result, dict) else {"text": str(result)}
        if "meta" not in payload:
            payload["meta"] = metadata
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return

    segments = result.get("segments")
    if not segments:
        raise ValueError("No segments returned; cannot build subtitles.")

    if output_format == "srt":
        output_path.write_text(build_srt(segments), encoding="utf-8")
        return

    if output_format == "vtt":
        output_path.write_text(build_vtt(segments), encoding="utf-8")
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def build_metadata(
    *,
    audio_path: Path,
    model: str,
    elapsed: float,
    output_format: str,
    language: str | None,
    task: str | None,
) -> dict[str, Any]:
    return {
        "audio_path": str(audio_path),
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "output_format": output_format,
        "language": language,
        "task": task,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def resolve_output_path(
    *,
    audio_path: Path,
    output_dir: Path | None,
    output_format: str,
) -> Path:
    target_dir = output_dir or audio_path.parent
    return target_dir / f"{audio_path.stem}.{output_format}"


def filter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if mlx_whisper is None:
        return kwargs
    try:
        signature = inspect.signature(mlx_whisper.transcribe)
    except (TypeError, ValueError):
        return kwargs
    supported = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in supported}


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if args.chunk_length is not None and args.chunk_length <= 0:
        errors.append("--chunk-length must be > 0.")
    if args.stride_length is not None and args.stride_length < 0:
        errors.append("--stride-length must be >= 0.")
    if args.batch_size is not None and args.batch_size <= 0:
        errors.append("--batch-size must be > 0.")
    if args.beam_size is not None and args.beam_size <= 0:
        errors.append("--beam-size must be > 0.")
    if args.temperature is not None and args.temperature < 0:
        errors.append("--temperature must be >= 0.")
    if args.compression_ratio_threshold is not None and args.compression_ratio_threshold <= 0:
        errors.append("--compression-ratio-threshold must be > 0.")
    if args.no_speech_threshold is not None and not (0 <= args.no_speech_threshold <= 1):
        errors.append("--no-speech-threshold must be between 0 and 1.")
    if args.logprob_threshold is not None and args.logprob_threshold > 0:
        errors.append("--logprob-threshold should be <= 0.")
    if args.output_format not in SUPPORTED_FORMATS:
        errors.append(f"Unsupported output format: {args.output_format}")
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    argv_list = list(argv or sys.argv[1:])
    config_path = find_config_path(argv_list)
    if config_path is None:
        default_config = Path.cwd() / DEFAULT_CONFIG_NAME
        config_path = default_config if default_config.exists() else None

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error reading config: {exc}")
        return EXIT_USAGE

    args = parse_args(argv_list, normalize_config(config))
    configure_logging(args.verbose, args.quiet)
    validation_errors = validate_args(args)
    for error in validation_errors:
        LOGGER.error(error)
    if validation_errors:
        return EXIT_USAGE

    if args.stdout and len(args.audio) > 1:
        LOGGER.error("--stdout only supports a single input file.")
        return EXIT_USAGE

    if mlx_whisper is None:
        LOGGER.error("mlx-whisper is not installed. Install it with `pip install mlx-whisper`.")
        return EXIT_USAGE

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        args.cache_dir = args.cache_dir.expanduser().resolve()
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Using cache dir: %s", args.cache_dir)

    exit_code = EXIT_OK
    total = len(args.audio)
    for index, audio in enumerate(args.audio, start=1):
        audio_path = Path(audio).expanduser().resolve()
        if not audio_path.is_file():
            LOGGER.error("File not found at %s", audio_path)
            exit_code = EXIT_RUNTIME
            continue

        LOGGER.info("Transcribing (%s/%s): %s", index, total, audio_path)
        LOGGER.info("Using model: %s", args.model)

        kwargs: dict[str, Any] = {"path_or_hf_repo": args.model}
        if args.language:
            kwargs["language"] = args.language
        if args.task:
            kwargs["task"] = args.task
        if args.chunk_length is not None:
            kwargs["chunk_length"] = args.chunk_length
        if args.stride_length is not None:
            kwargs["stride_length"] = args.stride_length
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        if args.beam_size is not None:
            kwargs["beam_size"] = args.beam_size
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.compression_ratio_threshold is not None:
            kwargs["compression_ratio_threshold"] = args.compression_ratio_threshold
        if args.logprob_threshold is not None:
            kwargs["logprob_threshold"] = args.logprob_threshold
        if args.no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = args.no_speech_threshold
        if args.word_timestamps:
            kwargs["word_timestamps"] = True
        if args.cache_dir:
            kwargs["cache_dir"] = str(args.cache_dir)

        kwargs = filter_kwargs(kwargs)

        start = time.perf_counter()
        try:
            result = mlx_whisper.transcribe(str(audio_path), **kwargs)
        except Exception as e:
            LOGGER.error("Error during transcription: %s", e)
            exit_code = EXIT_RUNTIME
            continue
        elapsed = time.perf_counter() - start

        if args.stdout:
            text = str(result.get("text", "")).strip()
            print(text)
            LOGGER.info("Transcription time: %s", format_elapsed(elapsed))
            return EXIT_OK

        output_path = resolve_output_path(
            audio_path=audio_path,
            output_dir=output_dir,
            output_format=args.output_format,
        )
        metadata = build_metadata(
            audio_path=audio_path,
            model=args.model,
            elapsed=elapsed,
            output_format=args.output_format,
            language=args.language,
            task=args.task,
        )

        try:
            write_output(
                output_path=output_path,
                output_format=args.output_format,
                result=result,
                overwrite=args.overwrite,
                metadata=metadata,
            )
            if args.metadata:
                meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
                meta_path.write_text(
                    json.dumps(metadata, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
        except Exception as e:
            LOGGER.error("Error writing transcript file: %s", e)
            exit_code = EXIT_RUNTIME
            continue

        LOGGER.info("Done! Transcript saved to: %s", output_path)
        LOGGER.info("Transcription time: %s", format_elapsed(elapsed))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
