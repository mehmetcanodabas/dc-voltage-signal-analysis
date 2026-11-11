## dc-voltage-signal-analysis

Python-Based Signal Analysis and Measurement Artifact Detection in a 110 V DC Substation Supply System

The project simulates a low-voltage alarm investigation in an electrical substation, inspired by practical experience in the energy sector.

FFT, Median Absolute Deviation (MAD), and median filtering are used to analyze voltage measurements, detect anomalies, and distinguish physical voltage drops from measurement artifacts through independent channel validation.

### Key Results

- 100 Hz voltage ripple identified using FFT.
- Three anomalies detected: one spike and two voltage drops.
- Isolated measurement spike corrected while preserving the physical voltage drop.
- Two low-voltage alarms investigated; one confirmed as a measurement-related false alarm.

### Result Snapshots

- [Figure 1 – DC Voltage Signal and Detected Anomalies](docs/results/Figure%201%20-%20DC%20Voltage%20Signal%20and%20Detected%20Anomalies.png): Simulated DC voltage measurements and anomalies.
- [Figure 2 – Frequency Analysis and Robust Anomaly Detection](docs/results/Figure%202%20-%20Frequency%20Analysis%20and%20Robust%20Anomaly%20Detection.png): FFT and MAD-based anomaly detection.
- [Figure 3 – Physical Voltage Drop vs Sensor Measurement Error](docs/results/Figure%203%20-%20Physical%20Voltage%20Drop%20vs%20Sensor%20Measurement%20Error.png): Independent measurement comparison.
- [Figure 4 – Spike Correction and Cross-Channel Validation](docs/results/Figure%204%20-%20Spike%20Correction%20and%20Cross-Channel%20Validation.png): Spike correction and measurement channel differences.

**Keywords:** Python, Anomaly Detection, DC Voltage, FFT, MAD, Measurement Artifacts, SCADA, Signal Processing.

**Data Source:** Synthetic voltage measurements.
