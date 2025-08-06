from unittest.mock import patch

import pytest
from ptwm.cli.main import main


def test_main_help_exists(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out, err = capsys.readouterr()
    assert "ptwm" in out


def test_compress_routing(tmp_path):
    # Test directory routing
    dir_path = tmp_path / "model_dir"
    dir_path.mkdir()

    with patch("ptwm.cli.compress.handle_compress") as mock_compress:
        main(["compress", str(dir_path)])
        mock_compress.assert_called_once()


def test_decompress_routing(tmp_path):
    file_path = tmp_path / "model.ptwm"
    file_path.touch()

    with patch("ptwm.cli.decompress.handle_decompress") as mock_decompress:
        main(["decompress", str(file_path)])
        mock_decompress.assert_called_once()
