"""mixarc CLI: arg parsing and server delegation (no real server started)."""
from unittest.mock import patch

from src.mixarc.cli import build_parser


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.port == 8770
    assert args.host == "127.0.0.1"
    assert args.no_browser is False
    assert args.dev is False
    assert args.config is None


def test_parser_config_flag():
    args = build_parser().parse_args(["--config", "/tmp/x.yaml", "--no-browser"])
    assert args.config == "/tmp/x.yaml"
    assert args.no_browser is True


def test_main_delegates_to_run_server():
    with patch("src.mixarc.cli.run_server") as rs:
        from src.mixarc.cli import main
        main(["--port", "9999", "--no-browser"])
    rs.assert_called_once()
    kw = rs.call_args.kwargs
    assert kw["port"] == 9999 and kw["open_browser"] is False
