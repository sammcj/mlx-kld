"""Tests for mlx_kld.cli helpers — pure-Python, no MLX/Metal needed."""

from __future__ import annotations

from mlx_kld.cli import _filter_dflash


def test_dflash_filter_drops_dflash_variants(capsys):
    paths = [
        "/models/foo/Qwen3.6-35B-A3B-8bit",
        "/models/Qwen3.6-35B-A3B-DFlash",
        "/models/bar/some-dflash-model",
        "/models/baz/Qwen3.6-27B-4bit",
    ]
    kept = _filter_dflash(paths, kind="--compare")
    assert kept == [
        "/models/foo/Qwen3.6-35B-A3B-8bit",
        "/models/baz/Qwen3.6-27B-4bit",
    ]
    err = capsys.readouterr().err
    assert "DFlash" in err
    assert "Qwen3.6-35B-A3B-DFlash" in err
    assert "some-dflash-model" in err


def test_dflash_filter_case_insensitive():
    paths = ["/models/X-DFLASH", "/models/Y-dFlAsH", "/models/Z-normal"]
    kept = _filter_dflash(paths, kind="--compare")
    assert kept == ["/models/Z-normal"]


def test_dflash_filter_no_match_returns_unchanged(capsys):
    paths = ["/models/a", "/models/b-4bit", "/models/c-8bit"]
    kept = _filter_dflash(paths, kind="--compare")
    assert kept == paths
    # No stderr note when nothing was filtered.
    assert capsys.readouterr().err == ""


def test_dflash_filter_empty_list():
    assert _filter_dflash([], kind="--compare") == []
