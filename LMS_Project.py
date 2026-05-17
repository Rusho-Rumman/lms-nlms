import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import pandas as pd

# =========================
# Load and Normalize Audio
# =========================
def load_and_normalize(filename):
    fs, data = wavfile.read(filename)

    if data.ndim > 1:
        data = data[:, 0]

    data = data.astype(np.float32)

    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    return fs, data


# =========================
# Save Audio
# =========================
def save_audio(filename, fs, signal):
    signal = signal / np.max(np.abs(signal))
    wavfile.write(filename, fs, (signal * 32767).astype(np.int16))


# =========================
# LMS Filter
# =========================
def lms_filter(primary, reference, mu, L):
    N = len(primary)

    w = np.zeros(L)
    y = np.zeros(N)   # estimated noise
    e = np.zeros(N)   # enhanced speech

    ref_padded = np.pad(reference, (L - 1, 0), mode='constant')

    for n in range(N):
        x_vec = ref_padded[n:n + L][::-1]

        y[n] = np.dot(w, x_vec)
        e[n] = primary[n] - y[n]

        w = w + mu * e[n] * x_vec

    return e, y, w


# =========================
# NLMS Filter
# =========================
def nlms_filter(primary, reference, mu, L, epsilon=1e-6):
    N = len(primary)

    w = np.zeros(L)
    y = np.zeros(N)
    e = np.zeros(N)

    ref_padded = np.pad(reference, (L - 1, 0), mode='constant')

    for n in range(N):
        x_vec = ref_padded[n:n + L][::-1]

        y[n] = np.dot(w, x_vec)
        e[n] = primary[n] - y[n]

        norm_factor = np.dot(x_vec, x_vec) + epsilon
        w = w + (mu / norm_factor) * e[n] * x_vec

    return e, y, w


# =========================
# Evaluation Metrics
# =========================
def calculate_snr(clean, test):
    noise = clean - test
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)

    return 10 * np.log10(signal_power / (noise_power + 1e-10))


def calculate_mse(clean, test):
    return np.mean((clean - test) ** 2)


# =========================
# Plot Results
# =========================
def plot_results(clean, noisy, lms_out, nlms_out, fs, true_noise, lms_noise, nlms_noise):
    time = np.arange(len(clean)) / fs

    # Waveform comparison
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time, clean)
    plt.title("Clean Speech s[n]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(time, noisy)
    plt.title("Noisy Speech x[n]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time, lms_out)
    plt.title("LMS Enhanced Speech e[n]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(time, nlms_out)
    plt.title("NLMS Enhanced Speech e[n]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Convergence plot
    # Better Convergence Plot
    lms_mse = (true_noise - lms_noise) ** 2
    nlms_mse = (true_noise - nlms_noise) ** 2

    # Moving average smoothing
    window_size = 500
    window = np.ones(window_size) / window_size

    lms_mse_smooth = np.convolve(lms_mse, window, mode='valid')
    nlms_mse_smooth = np.convolve(nlms_mse, window, mode='valid')

    # Convert sample index to time
    time_axis = np.arange(len(lms_mse_smooth)) / fs

    plt.figure(figsize=(12, 6))

    plt.plot(time_axis, 10 * np.log10(lms_mse_smooth + 1e-10),
            linewidth=2.2, label="LMS")

    plt.plot(time_axis, 10 * np.log10(nlms_mse_smooth + 1e-10),
            linewidth=2.2, label="NLMS")

    plt.title("Learning Curve / Convergence Comparison", fontsize=15)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Smoothed Noise Estimation Error (dB)", fontsize=12)

    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Spectrograms
    signals = [
        ("Clean Speech", clean),
        ("Noisy Speech", noisy),
        ("LMS Enhanced Speech", lms_out),
        ("NLMS Enhanced Speech", nlms_out)
    ]

    for title, sig in signals:
        f, t, Sxx = spectrogram(sig, fs)

        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.title("Spectrogram: " + title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power/Frequency (dB)")
        plt.tight_layout()
        plt.show()


# =========================
# Main Simulation
# =========================
def run_simulation():
    clean_file = "sp17.wav"
    noisy_file = "sp17_car_sn15.wav"

    fs_clean, clean_speech = load_and_normalize(clean_file)
    fs_noisy, noisy_speech = load_and_normalize(noisy_file)

    if fs_clean != fs_noisy:
        raise ValueError("Sampling frequencies do not match.")

    fs = fs_clean

    min_len = min(len(clean_speech), len(noisy_speech))
    clean_speech = clean_speech[:min_len]
    noisy_speech = noisy_speech[:min_len]

    # Actual noise from dataset
    true_noise = noisy_speech - clean_speech

    # Parameter testing
    mu_values = [0.001, 0.005, 0.01]
    L_values = [32, 64, 128]
    delay_values = [0, 5, 10]

    results = []

    best_lms_snr = -999
    best_nlms_snr = -999

    best_lms_output = None
    best_nlms_output = None
    best_lms_noise = None
    best_nlms_noise = None

    input_snr = calculate_snr(clean_speech, noisy_speech)
    input_mse = calculate_mse(clean_speech, noisy_speech)

    for mu in mu_values:
        for L in L_values:
            for delay in delay_values:

                reference_noise = np.roll(true_noise, delay)
                reference_noise[:delay] = 0

                lms_out, lms_noise, _ = lms_filter(noisy_speech, reference_noise, mu, L)
                nlms_out, nlms_noise, _ = nlms_filter(noisy_speech, reference_noise, mu, L)

                lms_snr = calculate_snr(clean_speech, lms_out)
                nlms_snr = calculate_snr(clean_speech, nlms_out)

                lms_mse = calculate_mse(clean_speech, lms_out)
                nlms_mse = calculate_mse(clean_speech, nlms_out)

                results.append([
                    mu, L, delay,
                    input_snr,
                    lms_snr,
                    lms_snr - input_snr,
                    nlms_snr,
                    nlms_snr - input_snr,
                    input_mse,
                    lms_mse,
                    nlms_mse
                ])

                if lms_snr > best_lms_snr:
                    best_lms_snr = lms_snr
                    best_lms_output = lms_out
                    best_lms_noise = lms_noise

                if nlms_snr > best_nlms_snr:
                    best_nlms_snr = nlms_snr
                    best_nlms_output = nlms_out
                    best_nlms_noise = nlms_noise

    columns = [
        "mu", "Filter Length L", "Delay",
        "Input SNR dB",
        "LMS Output SNR dB",
        "LMS Improvement dB",
        "NLMS Output SNR dB",
        "NLMS Improvement dB",
        "Input MSE",
        "LMS MSE",
        "NLMS MSE"
    ]

    results_df = pd.DataFrame(results, columns=columns)

    print("\n===== Performance Results =====")
    print(results_df)

    results_df.to_csv("ANC_LMS_NLMS_results.csv", index=False)

    print("\nBest LMS SNR:", round(best_lms_snr, 2), "dB")
    print("Best NLMS SNR:", round(best_nlms_snr, 2), "dB")

    # Save enhanced audio
    save_audio("lms_enhanced_speech.wav", fs, best_lms_output)
    save_audio("nlms_enhanced_speech.wav", fs, best_nlms_output)

    print("\nSaved files:")
    print("lms_enhanced_speech.wav")
    print("nlms_enhanced_speech.wav")
    print("ANC_LMS_NLMS_results.csv")

    # Plot best results
    plot_results(
        clean_speech,
        noisy_speech,
        best_lms_output,
        best_nlms_output,
        fs,
        true_noise,
        best_lms_noise,
        best_nlms_noise
    )


if __name__ == "__main__":
    run_simulation()