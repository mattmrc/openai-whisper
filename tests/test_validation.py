from whisper_cli import cli as run_whisper


def test_validate_args_rejects_bad_values():
    args = run_whisper.parse_args(["file.m4a"], {})
    args.chunk_length = -1
    args.no_speech_threshold = 2
    errors = run_whisper.validate_args(args)
    assert any("--chunk-length" in err for err in errors)
    assert any("--no-speech-threshold" in err for err in errors)
