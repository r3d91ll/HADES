from core.logging.conveyance import ConveyanceContext
from core.runtime.memgpt.boundary_metrics import BoundaryInputs, BoundaryMetricsCalculator


def test_boundary_metrics_basic():
    context = ConveyanceContext(L=0.8, I=0.9, A=0.85, G=0.8)
    inputs = BoundaryInputs(
        W_out=0.9,
        R_encode=0.95,
        H_out=1.0,
        T_out=0.4,
        W_in=0.85,
        R_decode=0.9,
        H_in=1.0,
        T_in=0.5,
        C_ext=context,
        P_ij=0.95,
        alpha=1.7,
    )

    calculator = BoundaryMetricsCalculator()
    payload = calculator.build_payload("turn-1", inputs)

    assert payload["label"] == "turn-1"
    assert payload["context"]["C_ext"] > 0
    assert payload["C_pair"] > 0
