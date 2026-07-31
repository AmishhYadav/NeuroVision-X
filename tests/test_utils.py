"""Tests for neurovision.utils: seed, device, logging, io.

All tests run on CPU, use tmp_path for any filesystem interaction, and avoid
depending on other tests' side effects. See CLAUDE.md for the project-wide
testing rules this suite follows.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neurovision.utils.device import amp_enabled, get_device
from neurovision.utils.io import ensure_dir, read_json, read_yaml, write_json, write_yaml
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

# --- shared fixtures ---


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Snapshot and restore root logger state so logging tests can't leak.

    setup_logging() mutates the global root logger (handlers + level). This
    ensures test ordering never lets one test's logging config bleed into
    another.
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in original_handlers:
        root.addHandler(handler)
    root.setLevel(original_level)


@pytest.fixture(autouse=True)
def _restore_cudnn_benchmark():
    """Restore the global cudnn.benchmark flag that set_seed mutates."""
    original = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.benchmark = original


# --- device ---


def test_get_device_cpu_string():
    device = get_device("cpu")
    assert device.type == "cpu"


def test_get_device_auto_never_mps():
    device = get_device("auto")
    assert device.type in {"cuda", "cpu"}


def test_get_device_from_config_object():
    cfg = SimpleNamespace(device="cpu")
    device = get_device(cfg)
    assert device.type == "cpu"


def test_get_device_case_and_whitespace_insensitive():
    device = get_device("  CPU  ")
    assert device.type == "cpu"


def test_get_device_bad_value_raises_value_error():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_get_device_missing_attribute_raises_attribute_error():
    cfg = SimpleNamespace()
    with pytest.raises(AttributeError):
        get_device(cfg)


def test_get_device_none_value_raises_value_error():
    cfg = SimpleNamespace(device=None)
    with pytest.raises(ValueError):
        get_device(cfg)


def test_amp_enabled_false_for_cpu():
    assert amp_enabled(torch.device("cpu")) is False


def test_amp_enabled_true_for_cuda_device_object():
    # Constructing a torch.device("cuda") does not require an actual GPU;
    # amp_enabled only inspects .type, so this is safe on a CPU-only machine.
    assert amp_enabled(torch.device("cuda")) is True


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only machine")
def test_get_device_cuda_unavailable_raises_runtime_error():
    with pytest.raises(RuntimeError):
        get_device("cuda")


# --- seed ---


def test_set_seed_returns_torch_generator():
    generator = set_seed(42)
    assert isinstance(generator, torch.Generator)


def test_set_seed_reproducible_torch():
    set_seed(0)
    first = torch.randn(4)
    set_seed(0)
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_set_seed_reproducible_python_random():
    set_seed(0)
    first = random.random()
    set_seed(0)
    second = random.random()
    assert first == second


def test_set_seed_reproducible_numpy():
    set_seed(0)
    first = np.random.rand()
    set_seed(0)
    second = np.random.rand()
    assert first == second


def test_set_seed_different_seeds_differ():
    set_seed(0)
    first = torch.randn(4)
    set_seed(1)
    second = torch.randn(4)
    assert not torch.equal(first, second)


def test_set_seed_returned_generator_is_reproducible():
    gen1 = set_seed(0)
    first = torch.randn(4, generator=gen1)
    gen2 = set_seed(0)
    second = torch.randn(4, generator=gen2)
    assert torch.equal(first, second)


def test_set_seed_restores_cudnn_benchmark_by_default():
    # MONAI's set_determinism sets cudnn.benchmark = False; set_seed puts it
    # back, because fixed-size patches make autotuning worth more than the
    # determinism it costs. The flag is readable on CPU-only machines.
    torch.backends.cudnn.benchmark = False
    set_seed(0)
    assert torch.backends.cudnn.benchmark is True


def test_set_seed_can_disable_cudnn_benchmark():
    set_seed(0, cudnn_benchmark=False)
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_negative_raises_value_error():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_set_seed_string_raises_type_error():
    with pytest.raises(TypeError):
        set_seed("42")


def test_set_seed_bool_raises_type_error():
    with pytest.raises(TypeError):
        set_seed(True)


# --- logging ---


def test_setup_logging_sets_root_level():
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_is_idempotent():
    setup_logging()
    setup_logging()
    stream_handlers = [
        h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1


def test_setup_logging_writes_to_nested_log_file(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "run.log"
    setup_logging("INFO", log_file=log_path)
    logger = logging.getLogger("test_setup_logging_writes_to_nested_log_file")
    logger.info("hello from test")
    # Close handlers so the file content is flushed to disk before reading.
    for handler in list(logging.getLogger().handlers):
        handler.close()
    assert log_path.is_file()
    assert "hello from test" in log_path.read_text(encoding="utf-8")


def test_setup_logging_invalid_level_raises_value_error():
    with pytest.raises(ValueError):
        setup_logging("NOT_A_LEVEL")


# --- io ---


def test_ensure_dir_creates_nested_path_and_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert target.is_dir()
    ensure_dir(target)  # calling again must not raise


def test_json_round_trip(tmp_path):
    path = tmp_path / "data.json"
    obj = {"a": 1, "b": [1, 2, 3]}
    write_json(obj, path)
    assert read_json(path) == obj


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "data.yaml"
    obj = {"a": 1, "b": [1, 2, 3]}
    write_yaml(obj, path)
    assert read_yaml(path) == obj


def test_write_json_creates_missing_parent_dir(tmp_path):
    path = tmp_path / "missing" / "dir" / "data.json"
    write_json({"a": 1}, path)
    assert path.is_file()


def test_write_yaml_creates_missing_parent_dir(tmp_path):
    path = tmp_path / "missing" / "dir" / "data.yaml"
    write_yaml({"a": 1}, path)
    assert path.is_file()


def test_write_json_serializes_path_via_default_str(tmp_path):
    path = tmp_path / "data.json"
    write_json({"p": tmp_path / "somewhere"}, path)
    result = read_json(path)
    assert result["p"] == str(tmp_path / "somewhere")


def test_read_json_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "does_not_exist.json")


def test_read_yaml_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "does_not_exist.yaml")


def test_read_yaml_empty_file_returns_none(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert read_yaml(path) is None
