"""Record method selection and extrapolation for an ethanol vapor pressure example."""
import json
import platform

import chemicals
import fluids
import thermo
from thermo import VaporPressure


def example_records():
    records = []
    # Fix the method explicitly: automatic rankings can change between releases.
    for extrapolation in (None, "Arrhenius"):
        inputs = dict(CASRN="64-17-5", method="WAGNER_MCGARRY",
                      extrapolation=extrapolation)
        psat = VaporPressure(**inputs)
        method = psat.method
        Tmin, Tmax = psat.T_limits[method]
        for T in (300.0, 100.0):
            within_limits = Tmin <= T <= Tmax
            value = psat.T_dependent_property(T)
            # For this fixed property/method, these are the evaluation paths.
            # None is a failure/unavailable result, never a numerical zero.
            if value is None:
                evaluation = "unavailable"
            elif within_limits:
                evaluation = "correlation"
            else:
                evaluation = "extrapolation"
            records.append(dict(
                constructor=inputs.copy(), property=psat.name, units=psat.units,
                T_K=T, method=method, T_limits_K=[Tmin, Tmax],
                within_limits=within_limits, evaluation=evaluation, value=value,
                versions=dict(python=platform.python_version(), thermo=thermo.__version__,
                              chemicals=chemicals.__version__, fluids=fluids.__version__),
            ))
    return records


if __name__ == "__main__":
    print(json.dumps(example_records(), indent=2, allow_nan=False))
