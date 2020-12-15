Introduction to Phase and Flash Calculations
============================================

I have programmed interfaces for doing property calculations at least four times now, and have settled on the current interface which is designed around the following principles:

* Immutability
* Calculations are completely independent from any databases or lookups - every input must be provided as input
* Inclusion of separate flashes algorithms wherever faster algorithms can be used for specific cases
* Default to general-purpose algorithms that make no assumptions about specific systems
* Use very tight tolerances on all calculations
* Expose all constants used by algorithms


After a couple of iterations, the interface which is found to work best for 

>>> constants = ChemicalConstantsPackage(Tcs=[768.0], Pcs=[1070000.0], omegas=[0.8805], MWs=[282.54748], CASs=['112-95-8'])