Recording a property calculation
================================

A result is easier to review when it is saved with its chemical identity,
units, inputs, selected method, temperature limits and extrapolation setting.
The example below records these for ethanol vapor pressure using existing
public APIs. It explicitly selects ``WAGNER_MCGARRY`` so changes to automatic
method rankings do not silently change the chosen correlation.

At 300 K the selected correlation is evaluated inside its limits. At 100 K,
extrapolation disabled with ``None`` gives an unavailable result, while
``Arrhenius`` enables an extrapolated result. The JSON record retains this
distinction. ``None`` is serialized as ``null``; it must not be replaced by
zero. In general, a ``None`` return may also indicate a calculation failure or
an invalid property value; it is not a diagnosis of the cause.

.. literalinclude:: property_calculation_audit.py
   :language: python

The script is also available for :download:`download <property_calculation_audit.py>`.
Run it with ``python property_calculation_audit.py`` to print the records.

Scope and interpretation
------------------------

* ``T_limits`` describes the range used by the property implementation. Being
  within that range does not establish experimental provenance or quantify
  uncertainty. A successful calculation is not an accuracy guarantee.
* A method identifier is a starting point for tracing a calculation, not a
  complete citation. Consult the selected method's description and references
  in :obj:`thermo.vapor_pressure.VaporPressure`, then inspect the underlying
  coefficient source and its references when that level of provenance is
  needed. Do not infer that data are experimental just from a method name.
* This example assumes the selected vapor-pressure method has temperature
  limits. Other property types or methods may expose different metadata;
  missing limits must not be interpreted as an unlimited validated range.
* Extrapolation is an additional model choice. Its output is deliberately
  labelled separately from an in-range correlation result.
* The constructor inputs and versions help reproduce this example. For
  modified installations, also retain the source revision, dependency lock
  file, and any custom coefficients or data. Versions alone do not identify
  locally changed data. Pressure-dependent and mixture calculations also
  require pressure, composition, and the relevant method selections.

This is a calculation record, not a general experimental-data classifier or
an uncertainty model. See :doc:`../property_objects` for the underlying API.
