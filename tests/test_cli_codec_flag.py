import pytest
from ptwm.cli.main import main


def test_compress_help_lists_codec_and_device(capsys):
    with pytest.raises(SystemExit):
        main(["compress", "--help"])
    out, _ = capsys.readouterr()
    assert "--codec" in out
    assert "--device" in out


def test_decompress_help_lists_device(capsys):
    with pytest.raises(SystemExit):
        main(["decompress", "--help"])
    out, _ = capsys.readouterr()
    assert "--device" in out
