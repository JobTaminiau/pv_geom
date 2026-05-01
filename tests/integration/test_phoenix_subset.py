"""Phoenix integration smoke test — implemented in M7. Gated on RUN_INTEGRATION env var."""

import os

import pytest

pytestmark = pytest.mark.integration

if not os.environ.get("RUN_INTEGRATION"):
    pytest.skip("set RUN_INTEGRATION=1 to run the Phoenix smoke test", allow_module_level=True)


def test_phoenix_smoke():
    pytest.skip("M7: implement after the runner lands")
