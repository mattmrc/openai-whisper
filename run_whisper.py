import sys
import time
from pathlib import Path
import mlx_whisper

# Options:
# "mlx-community/whisper-base-mlx"
# "mlx-community/whisper-large-v3-turbo"
# "mlx-community/whisper-large-v3-mlx"
MODEL_NAME = "mlx-community/whisper-large-v3-turbo"


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


def main() -> int:
    if len(sys.argv) < 2:
        print("Error: No file provided.")
        print("Usage: python mlx_whisper_gemini.py /path/to/meeting.m4a")
        return 1

    audio_path = Path(sys.argv[1]).expanduser().resolve()

    if not audio_path.is_file():
        print(f"Error: File not found at {audio_path}")
        return 1

    print(f"Transcribing: {audio_path}")
    print(f"Using model: {MODEL_NAME}")

    start = time.perf_counter()
    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=MODEL_NAME,
        )
    except Exception as e:
        print(f"Error during transcription: {e}")
        return 1
    elapsed = time.perf_counter() - start

    text = result.get("text", "") if isinstance(result, dict) else str(result)
    text = text.strip()

    output_path = audio_path.with_suffix(".txt")

    try:
        output_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"Error writing transcript file: {e}")
        return 1

    print(f"Done! Transcript saved to: {output_path}")
    print(f"Transcription time: {format_elapsed(elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
