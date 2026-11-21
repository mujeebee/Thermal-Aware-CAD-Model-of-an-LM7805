## Overview

A linear regulator wastes the difference between its input and output voltage as heat. At 12 V in, 5 V out and
0.5 A, an LM7805 dissipates 3.5 W, enough to push its junction past 260 °C in free air. This project designs a
finned aluminium heat sink for it and checks that the design works.

- **Electronics:** the heat load comes from the circuit, P = (Vin − Vout) × I = 3.5 W.
- **CAD:** the PCB, regulator and 7-fin heat sink (40 × 40 mm base, 25 mm fins, 4.9 mm gaps) are modelled in
  Siemens NX.
- **Thermal analysis:** the junction temperature is predicted by hand (thermal resistance chain) and by a custom
  3D finite-volume solver in Python that runs on STL files exported from the NX model.

| | Result |
| --- | --- |
| Junction temperature (simulation) | 79.0 °C |
| Junction temperature (hand calculation) | 80.2 °C |
| Design limit | 100 °C |
| No heat sink | about 260-270 °C |

