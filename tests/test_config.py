from pathlib import Path

from whisper_cli import cli as run_whisper


def test_load_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "test-model"\nverbose = 1\n', encoding="utf-8")
    config = run_whisper.load_config(config_path)
    assert config["model"] == "test-model"
    assert config["verbose"] == 1
