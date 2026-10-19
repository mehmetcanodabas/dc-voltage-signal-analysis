import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, label
from scipy.signal import medfilt

# SIMULATED SUBSTATION DC AUXILIARY SUPPLY: ALARM INVESTIGATION
# This is a synthetic engineering case study, not a historical site record.
fs = 2000
T = 1.0
t = np.arange(0, T, 1 / fs)
N = len(t)
rng = np.random.default_rng(42)

nominal_dc = 110.0
ripple_frequency = 100.0
ripple_amplitude = 0.5
ripple = ripple_amplitude * np.sin(2 * np.pi * ripple_frequency * t)

# Ground-truth physical event: a 30 ms, 6.5 V reduction in DC supply.
physical_mask = (t >= 0.20) & (t < 0.23)
physical_drop = np.zeros(N)
physical_drop[physical_mask] = -6.5
true_bus_voltage = nominal_dc + ripple + physical_drop

# Primary SCADA-associated measurement, with ordinary additive noise.
primary_noise = rng.normal(0, 0.1, N)
primary_reference = true_bus_voltage + primary_noise
primary = primary_reference.copy()

# Two independent acquisition artifacts, injected ONLY in the primary channel.
spike_index = int(0.05 * fs)
primary[spike_index] = 113.0
sensor_mask = (t >= 0.40) & (t < 0.42)
primary[sensor_mask] -= 7.5

# A separate fast recorder is assumed to have been installed before the
# investigation capture. It measures the same bus with independent electronics.
secondary_noise = rng.normal(0, 0.1, N)
independent_recorder = true_bus_voltage + secondary_noise


def analyze_voltage(signal, sampling_rate, time):
    # The 0.10–0.20 s interval is deliberately designated as a stable
    # calibration interval for THIS simulated case. It has no injected event.
    # For unknown real records, that interval must first be validated.
    stable = (time >= 0.10) & (time < 0.20)
    segment = signal[stable]
    segment_t = time[stable]
    n = len(segment)
    dc = float(np.median(signal))
    fft = np.fft.rfft(segment - np.mean(segment))
    frequencies = np.fft.rfftfreq(n, d=1 / sampling_rate)
    amplitudes = 2 * np.abs(fft) / n
    amplitudes[0] /= 2
    if n % 2 == 0:
        amplitudes[-1] /= 2
    peak = np.argmax(amplitudes[1:]) + 1
    frequency = float(frequencies[peak])
    amplitude = float(amplitudes[peak])
    # FFT phase is relative to the calibration window start.
    phase = float(np.angle(fft[peak]) + np.pi / 2 - 2 * np.pi * frequency * segment_t[0])
    phase = float(np.arctan2(np.sin(phase), np.cos(phase)))
    estimated_ripple = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    residual = signal - dc - estimated_ripple
    return dict(dc=dc, frequency=frequency, amplitude=amplitude,
                phase=phase, ripple=estimated_ripple,
                residual=residual, residual_std=float(np.std(residual, ddof=1)),
                frequencies=frequencies, spectrum=amplitudes)


def contiguous_events(mask):
    """Return (start, stop-exclusive) ranges for True runs."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def low_voltage_alarm(signal, threshold, minimum_duration_ms, sampling_rate):
    """Illustrative fast-acquisition alarm with persistence (not actual SCADA timing)."""
    under_threshold = signal < threshold
    minimum_samples = int(np.ceil(minimum_duration_ms * sampling_rate / 1000))
    confirmed = np.zeros_like(under_threshold, dtype=bool)
    for start, stop in contiguous_events(under_threshold):
        if stop - start >= minimum_samples:
            confirmed[start:stop] = True
    return confirmed


before = analyze_voltage(primary, fs, t)

# Robust residual-based anomaly detection.
center = float(np.median(before['residual']))
mad = float(np.median(np.abs(before['residual'] - center)))
robust_sigma = 1.4826 * mad
threshold = 6.0 * robust_sigma
anomaly_mask = np.abs(before['residual'] - center) > threshold

# Group nearby anomalous samples and use event duration only as a first pass.
grouped = binary_closing(anomaly_mask, structure=np.ones(7, dtype=bool))
event_labels, event_count = label(grouped)
sustained_mask = np.zeros(N, dtype=bool)
events = []
for event_id in range(1, event_count + 1):
    indices = np.flatnonzero(event_labels == event_id)
    start, stop = indices[0], indices[-1] + 1
    sustained = len(indices) >= 5
    if sustained:
        sustained_mask[indices] = True
    events.append((start, stop, sustained))

# First-pass correction: isolate and suppress a single-sample candidate.
isolated_mask = anomaly_mask & ~sustained_mask
median_estimate = medfilt(primary, kernel_size=5)
corrected = primary.copy()
corrected[isolated_mask] = median_estimate[isolated_mask]
after = analyze_voltage(corrected, fs, t)

# Compare both measurements. A large mismatch suggests a channel-specific
# problem; agreement suggests, but does not prove, a physical bus event.
channel_difference = corrected - independent_recorder
mismatch_limit = 1.0  # V; illustrative engineering tolerance
classified_events = []
for start, stop, sustained in events:
    mean_difference = float(np.mean(channel_difference[start:stop]))
    if not sustained:
        result = 'ISOLATED MEASUREMENT ARTIFACT CANDIDATE'
    elif abs(mean_difference) > mismatch_limit:
        result = 'CHANNEL-SPECIFIC MEASUREMENT ERROR CANDIDATE'
    else:
        result = 'COMMON PHYSICAL EVENT CANDIDATE'
    classified_events.append((start, stop, result, mean_difference))

# Example alarm logic operating on a fast 2 kHz channel. An actual RTU/SCADA
# may sample, filter, or delay alarms differently; validate configuration.
alarm_threshold = 105.0
alarm_persistence_ms = 10.0
primary_alarm = low_voltage_alarm(primary, alarm_threshold, alarm_persistence_ms, fs)
recorder_alarm = low_voltage_alarm(independent_recorder, alarm_threshold, alarm_persistence_ms, fs)
corrected_alarm = low_voltage_alarm(corrected, alarm_threshold, alarm_persistence_ms, fs)

print('\n--- Signal Analysis ---')
print(f'Sampling rate: {fs} Hz; N: {N}')
print(f'Estimated dominant ripple: {before["frequency"]:.1f} Hz, {before["amplitude"]:.4f} V')
print(f'MAD: {mad:.5f} V; robust sigma: {robust_sigma:.5f} V')
print(f'Adaptive anomaly threshold: {threshold:.5f} V')
print(f'Flagged samples: {np.sum(anomaly_mask)}')
print(f'Samples corrected (isolated only): {np.sum(isolated_mask)}')
print(f'Physical-event samples modified: {np.sum(corrected[physical_mask] != primary[physical_mask])}')
print(f'Mean absolute error vs primary noise-containing reference, before: {np.mean(np.abs(primary - primary_reference)):.5f} V')
print(f'Mean absolute error vs primary noise-containing reference, after:  {np.mean(np.abs(corrected - primary_reference)):.5f} V')
print(f'Residual std, before: {before["residual_std"]:.4f} V; after: {after["residual_std"]:.4f} V')

print('\n--- Event Investigation ---')
for i, (start, stop, category, difference) in enumerate(classified_events, 1):
    print(f'Event {i}: {t[start]:.4f}–{t[stop-1]:.4f} s; '
          f'{(stop-start)/fs*1000:.1f} ms; '
          f'mean channel difference={difference:+.3f} V; {category}')

print('\n--- Illustrative Low-Voltage Alarms ---')
print(f'Alarm threshold: < {alarm_threshold:.1f} V for at least {alarm_persistence_ms:.1f} ms')
for name, mask in [('Primary SCADA channel', primary_alarm),
                   ('Independent recorder', recorder_alarm),
                   ('Spike-corrected primary', corrected_alarm)]:
    ranges = contiguous_events(mask)
    descriptions = [f'{t[s]:.3f}–{t[e-1] + 1/fs:.3f} s' for s, e in ranges]
    print(f'{name}: {len(ranges)} event(s): {descriptions}')

print('\n--- Ground-Truth Verification (Simulation Only) ---')
print(f'Primary sensor fault mean bias: {np.mean(primary[sensor_mask] - primary_reference[sensor_mask]):+.3f} V')
print(f'After isolated-spike correction, sensor fault bias: {np.mean(corrected[sensor_mask] - primary_reference[sensor_mask]):+.3f} V')
print('Ground truth is used for validation, NOT supplied to the detection algorithm.')

# 1: Full waveform and alarm events.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t, primary, label='Primary / SCADA-linked channel', linewidth=1)
ax.plot(t, independent_recorder, label='Independent fast recorder', alpha=0.8, linewidth=1)
ax.axhline(alarm_threshold, color='red', ls='--', label='Illustrative low-voltage threshold (105 V)')
ax.axvspan(0.20, 0.23, color='green', alpha=0.1, label='Physical event (truth)')
ax.axvspan(0.40, 0.42, color='orange', alpha=0.12, label='Sensor error (truth)')
ax.set(xlabel='Time (s)', ylabel='Voltage (V)', title='110 V DC Supply: Three Anomalies')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True)
fig.tight_layout()

# 2: FFT and residual outliers.
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(before['frequencies'], before['spectrum'])
axes[0].set(xlim=(0, 300), xlabel='Frequency (Hz)', ylabel='Amplitude (V)', title='FFT of Primary Channel')
axes[0].grid(True)
axes[1].plot(t, before['residual'], label='Residual', linewidth=1)
axes[1].scatter(t[anomaly_mask], before['residual'][anomaly_mask], s=9, color='red', label='MAD anomaly candidates')
axes[1].axhline(center + threshold, ls='--', color='black')
axes[1].axhline(center - threshold, ls='--', color='black')
axes[1].set(xlabel='Time (s)', ylabel='Residual (V)', title='Robust Anomaly Detection (MAD)')
axes[1].legend()
axes[1].grid(True)
fig.tight_layout()

# 3: Both sustained events, visually compared against an independent channel.
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for ax, start, stop, title in [
    (axes[0], 0.18, 0.25, 'Physical Voltage Drop: Both Channels Agree'),
    (axes[1], 0.38, 0.44, 'Sensor Measurement Error: Channels Disagree')]:
    ax.plot(t, primary, label='Primary measurement')
    ax.plot(t, independent_recorder, label='Independent recorder', ls='--')
    ax.axhline(alarm_threshold, color='red', ls=':', label='Low-voltage threshold')
    ax.set(xlim=(start, stop), xlabel='Time (s)', ylabel='Voltage (V)', title=title)
    ax.grid(True)
    ax.legend(fontsize=8)
fig.tight_layout()

# 4: Illustrate correction and remaining error.
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(t, primary, label='Before correction')
axes[0].plot(t, corrected, ls='--', label='After isolated-spike correction')
axes[0].plot(t, independent_recorder, alpha=0.75, label='Independent recorder')
axes[0].set(xlim=(0.04, 0.06), xlabel='Time (s)', ylabel='Voltage (V)', title='Spike: Single-Sample Correction')
axes[0].legend()
axes[0].grid(True)
axes[1].plot(t, channel_difference, label='Corrected primary - independent recorder')
axes[1].axhline(mismatch_limit, color='red', ls='--')
axes[1].axhline(-mismatch_limit, color='red', ls='--')
axes[1].set(xlabel='Time (s)', ylabel='Channel difference (V)', title='Cross-Channel Validation (Not Blind Signal Cleaning)')
axes[1].legend()
axes[1].grid(True)
fig.tight_layout()

plt.show()
