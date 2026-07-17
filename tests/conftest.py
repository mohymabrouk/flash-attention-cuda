"""Shared pytest configuration for CPU and optional CUDA verification."""

from __future__ import annotations

from typing import Final

import pytest


CUDA_MARKER: Final = "cuda"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flash-attention-cuda")
    group.addoption(
        "--require-cuda",
        action="store_true",
        default=False,
        help=(
            "Fail instead of skipping when CUDA or the compiled extension is unavailable. "
            "Use this on Kaggle and other GPU validation runners."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cuda: requires a CUDA device and the compiled flash-attention extension",
    )


def _cuda_unavailable_reason() -> str | None:
    try:
        import torch
    except ImportError as exc:
        return f"PyTorch could not be imported: {exc}"

    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is false"

    try:
        from flash_attention_cuda import extension_available, extension_error
    except ImportError as exc:
        return f"flash_attention_cuda could not be imported: {exc}"

    if not extension_available():
        detail = extension_error()
        return f"compiled extension is unavailable: {detail or 'unknown import error'}"
    return None


def pytest_sessionstart(session: pytest.Session) -> None:
    """Turn missing GPU prerequisites into an immediate error when requested."""

    if not session.config.getoption("--require-cuda"):
        return
    reason = _cuda_unavailable_reason()
    if reason is not None:
        raise pytest.UsageError(f"--require-cuda was set, but {reason}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--require-cuda"):
        return
    if not any(item.get_closest_marker(CUDA_MARKER) is not None for item in items):
        raise pytest.UsageError(
            "--require-cuda was set, but no tests marked 'cuda' were collected"
        )


@pytest.fixture(scope="session")
def cuda_device(request: pytest.FixtureRequest):
    """Return the active CUDA device or skip the optional GPU suite cleanly."""

    import torch

    reason = _cuda_unavailable_reason()
    if reason is not None:
        if request.config.getoption("--require-cuda"):
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    return torch.device("cuda", torch.cuda.current_device())
