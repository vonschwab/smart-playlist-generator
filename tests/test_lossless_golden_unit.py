"""Unit tests for the golden capture encoder/decoder (no artifact needed)."""
import json

import pytest

from tests.support.lossless_golden import _decode, _json_default


def test_set_roundtrips_deterministically():
    encoded = json.dumps({"excluded": {"b", "a", "c"}}, default=_json_default)
    decoded = json.loads(encoded, object_hook=_decode)
    assert decoded["excluded"] == {"a", "b", "c"}


def test_non_serializable_raises():
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=_json_default)
