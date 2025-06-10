import subprocess


def test_cli_help():
    result = subprocess.run(
        ["poetry", "run", "obsidian-agent", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
