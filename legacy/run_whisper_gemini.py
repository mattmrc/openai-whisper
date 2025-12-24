import os
import sys

import mlx_whisper

# --- CONFIGURATION ---
# Options:
# "mlx-community/whisper-base-mlx"          (Fastest, lower accuracy)
# "mlx-community/whisper-large-v3-turbo"    (BEST BALANCE for meetings)
# "mlx-community/whisper-large-v3-mlx"      (Maximum accuracy, slower)
MODEL_NAME = "mlx-community/whisper-large-v3-turbo"


# ---------------------


def transcribe_audio():
    if len(sys.argv) < 2:
        print("❌ Error: No file provided.")
        print("Usage: python run_whisper_gemini.py /path/to/meeting.m4a")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"❌ Error: File not found at {audio_path}")
        sys.exit(1)

    print(f"🎧 Transcribing: {audio_path}")
    print(f"🧠 Using Model: {MODEL_NAME}")

    # Run transcription
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=MODEL_NAME)

    # Save file
    base_name = os.path.splitext(audio_path)[0]
    output_path = f"{base_name}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"✅ Done! Transcript saved to: {output_path}")


if __name__ == "__main__":
    transcribe_audio()
