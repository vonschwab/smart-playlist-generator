"""
Task 5 threading test: verify energy_matrix is wired through the builder entrypoint.
"""
import inspect

from src.playlist import pier_bridge_builder as b


def test_builder_accepts_energy_matrix_param():
    sig = inspect.signature(b.build_pier_bridge_playlist)
    assert "energy_matrix" in sig.parameters, (
        f"build_pier_bridge_playlist missing energy_matrix param; found: {list(sig.parameters)}"
    )
