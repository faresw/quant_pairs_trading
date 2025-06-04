import pandas as pd
from signal_generator import generate_signals


def test_generate_signals_exit_behaviour():
    z = pd.Series([0.0, 2.1, 2.3, 0.2, -2.2, -2.5, -0.1, 0.3])
    signals = generate_signals(z, entry_threshold=2.0, exit_threshold=0.5)
    expected = pd.Series([0, -1, -1, 0, 1, 1, 0, 0], name="position")
    pd.testing.assert_series_equal(signals["position"], expected)
