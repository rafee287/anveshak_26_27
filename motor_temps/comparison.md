# Temperature Sensor Comparison: TMP36 vs. NTC Thermistor

### Overview
* **TMP36:** An active semiconductor integrated circuit (IC) providing a linear voltage output.
* **NTC Thermistor:** A passive variable resistor whose resistance decreases exponentially as temperature rises.

---

### Specification Matrix


| Feature | TMP36 IC Sensor | NTC Thermistor (10kΩ Typical) |
| :--- | :--- | :--- |
| **Output Type** | Linear voltage ($10\text{ mV}/^\circ\text{C}$) | Non-linear resistance (exponential curve) |
| **Circuit Design** | Direct connection to ADC pin | Requires a voltage divider circuit |
| **Pin / Wire Count** | 3 pins ($V_{\text{CC}}$, $GND$, $V_{\text{OUT}}$) | 2 pins / wires (reversible orientation) |
| **Software Math** | Simple linear equation | Complex Steinhart-Hart equation |
| **Temperature Range**| $-40^\circ\text{C}$ to $+125^\circ\text{C}$ | $-55^\circ\text{C}$ to $+200^\circ\text{C}$+ (probe dependent) |
| **Supply Voltage Sensitivity** | Vulnerable to $V_{\text{CC}}$ fluctuations | Immune to $V_{\text{CC}}$ fluctuations (ratiometric) |
| **Form Factors** | Plastic IC packages (TO-92, SOT-23) | Glass beads, epoxy beads, metal probes, threads |
| **Relative Cost** | Higher (microchip architecture) | Extremely low (basic raw component) |

---

### Deep Dive: Core Differences

#### 1. Linearity & Calibration
* **TMP36:** Factory calibrated. A change of $1^\circ\text{C}$ always yields exactly a $10\text{ mV}$ change. It requires zero calibration in software.
* **NTC Thermistor:** Highly non-linear. A change of $1^\circ\text{C}$ causes a large resistance shift at low temperatures, but a tiny resistance shift at high temperatures. Software must use natural logarithms to linearize the data.

#### 2. Circuit Architecture
* **TMP36:** Functions independently as an active device. It outputs an absolute voltage based on internal bandgap references.
* **NTC Thermistor:** Requires a static companion resistor (usually $10\text{k}\Omega$) to form a voltage divider. Because the microcontroller ADC reads a ratio rather than an absolute voltage, fluctuations in your power supply rail will not warp your temperature readings.

#### 3. Ruggedness & Environment
* **TMP36:** Delicately housed in plastic IC casings. It cannot be submerged in liquids or exposed to outdoor weather without custom waterproof shielding.
* **NTC Thermistor:** Readily available in pre-sealed, heavy-duty stainless steel housings. It is the industrial standard for liquid immersion, automotive engines, and 3D printer hotends.

---

### Target Use Cases

* **Use the TMP36 if:** We are building indoor prototypes, breadboarding hobby projects, or want to write clean code without dealing with logarithmic mathematical conversions.
* **Use the NTC Thermistor if:** We need a rugged sensor for harsh environments, need to measure temperatures exceeding $125^\circ\text{C}$, or are designing a commercial circuit board where component unit costs must be minimized.

---

### Conclusion 

As we need to measure motor temperatures, considering factors like response time, ease of mounting, sensitivity and cost we are going to go with the NTC thermistors, particularly the MF52 series.

