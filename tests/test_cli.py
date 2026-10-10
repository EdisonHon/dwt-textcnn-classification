from __future__ import annotations

from pathlib import Path

import pytest

from dwt_textcnn.cli import COMMAND_PHASES, build_parser, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_six_commands_are_registered() -> None:
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if action.dest == "command"
    )

    assert set(subparser_action.choices) == set(COMMAND_PHASES)
    assert len(subparser_action.choices) == 6


@pytest.mark.parametrize("command", COMMAND_PHASES)
def test_each_command_has_help(command: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0


def test_check_config_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "train",
            "--config",
            str(PROJECT_ROOT / "configs" / "original.yaml"),
            "--check-config",
        ]
    )

    assert result == 0
    assert '"method": "original"' in capsys.readouterr().out
