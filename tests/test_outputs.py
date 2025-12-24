from pathlib import Path

from whisper_cli import cli as run_whisper


def test_build_srt_and_vtt():
    segments = [
        {"start": 0.0, "end": 1.25, "text": "Hello"},
        {"start": 1.25, "end": 2.5, "text": "World"},
    ]
    srt = run_whisper.build_srt(segments)
    vtt = run_whisper.build_vtt(segments)
    assert "1" in srt
    assert "00:00:00,000 --> 00:00:01,250" in srt
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:01.250" in vtt


def test_write_output_json_includes_metadata(tmp_path: Path):
    result = {"text": "Hello"}
    metadata = {"model": "test-model"}
    output_path = tmp_path / "out.json"
    run_whisper.write_output(
        output_path=output_path,
        output_format="json",
        result=result,
        overwrite=True,
        metadata=metadata,
    )
    payload = output_path.read_text(encoding="utf-8")
    assert "\"meta\"" in payload


def test_write_output_txt(tmp_path: Path):
    result = {"text": "Hello"}
    output_path = tmp_path / "out.txt"
    run_whisper.write_output(
        output_path=output_path,
        output_format="txt",
        result=result,
        overwrite=True,
        metadata={},
    )
    assert output_path.read_text(encoding="utf-8").strip() == "Hello"
