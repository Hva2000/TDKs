import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import librosa
import librosa.display
from scipy.signal import find_peaks, butter, filtfilt
import csv

# paraméterek
rho = 7850
d = 0.001067
L = 0.675
f1 = 82.41
A = np.pi * (d / 2)**2
mu = rho * A
T = (2 * L * f1)**2 * mu
c = np.sqrt(T / mu)

print(f"[INFO] Feszítőerő: {T:.2f} N")
print(f"[INFO] Hullámsebesség: {c:.2f} m/s")

#diszkretizálás
Nx = 200
dx = L / (Nx - 1)
dt = 0.25 * dx / c
t_max = 0.5
Nt = int(t_max / dt)
x = np.linspace(0, L, Nx)
y = np.zeros(Nx)
y_old = np.zeros(Nx)
y[int(Nx // 2)] = 0.01
y_old[:] = y[:]
frames = []

#sizimulációs
for n in range(Nt):
    y_new = np.zeros(Nx)
    for i in range(1, Nx - 1):
        y_new[i] = 2 * y[i] - y_old[i] + (c * dt / dx)**2 * (y[i + 1] - 2 * y[i] + y[i - 1])
    y_old[:] = y[:]
    y[:] = y_new[:]
    if n % 10 == 0:
        frames.append(y.copy())

#WAV save
middle = np.array([frame[Nx // 2] for frame in frames])
middle = middle / np.max(np.abs(middle))
sr = int(1 / (dt * 10))
write("simulated.wav", sr, (middle * 32767).astype(np.int16))
print("[INFO] Modellezett .wav fájl mentve.")

#Spektrum fügv készítése
def compute_spectrum(signal, sr, label):
    D = np.abs(librosa.stft(signal, n_fft=8192))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    spectrum = np.mean(D, axis=1)
    base_idx = np.argmin(np.abs(freqs - f1))
    base_amp = spectrum[base_idx]
    spectrum = spectrum / base_amp if base_amp > 0 else spectrum
    peaks, _ = find_peaks(spectrum, height=0.05)
    peak_freqs = freqs[peaks]
    peak_amps = spectrum[peaks]
    print(f"\n[{label}] Top 10 harmonikus:")
    for i in range(min(10, len(peak_freqs))):
        print(f"{i+1}. {peak_freqs[i]:.2f} Hz - Amp: {peak_amps[i]:.2f}")
    return freqs, spectrum

#szűrések
def bandpass_filter(signal, lowcut=60, highcut=2000, fs=44100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

#valós mikrofonos felvétel
y_real, sr_real = librosa.load("izolalt.wav", sr=None)
y_real = bandpass_filter(y_real, fs=sr_real)
y_real *= 5.0
y_real = y_real / np.max(np.abs(y_real))
freq_real, spec_real = compute_spectrum(y_real, sr_real, "Izolált húr (mikrofon)")

# modellezet spektrum
y_sim, sr_sim = librosa.load("simulated.wav", sr=None)
freq_sim, spec_sim = compute_spectrum(y_sim, sr_sim, "Modellezett húr")

# les paulos pickup felvétel
y_pickup, sr_pickup = librosa.load("pickup.wav", sr=None)
y_pickup = bandpass_filter(y_pickup, fs=sr_pickup)
y_pickup *= 5.0
y_pickup = y_pickup / np.max(np.abs(y_pickup))
freq_pickup, spec_pickup = compute_spectrum(y_pickup, sr_pickup, "Gitár pickup")

# ábrák
plt.figure(figsize=(12, 6))
plt.plot(freq_sim, spec_sim, label="Modellezett húr", linewidth=2, alpha=0.8)
plt.plot(freq_real, spec_real, label="Izolált húr (mikrofon)", linewidth=2, alpha=0.8)
plt.plot(freq_pickup, spec_pickup, label="Gitár pickup", linewidth=2, alpha=0.8)
plt.title("Modellezett – Mikrofon – Gitár pickup spektrum összehasonlítása")
plt.xlabel("Frekvencia [Hz]")
plt.ylabel("Normalizált amplitúdó")
plt.xlim(50, 2000)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("spektrum_3osszehasonlitas.png", dpi=300)
plt.show()

# txt export
def save_top_peaks_txt(freqs_sim, spec_sim, freqs_real, spec_real, filename="spektrum_osszehasonlitas.txt"):
    peaks_sim, _ = find_peaks(spec_sim, height=0.05)
    peaks_real, _ = find_peaks(spec_real, height=0.05)
    top_sim = sorted(zip(freqs_sim[peaks_sim], spec_sim[peaks_sim]), key=lambda x: -x[1])[:10]
    top_real = sorted(zip(freqs_real[peaks_real], spec_real[peaks_real]), key=lambda x: -x[1])[:10]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Domináns harmonikus frekvenciák összehasonlítása (modellezett vs. izolált)\n\n")
        f.write(f"{'Harm.':<6}{'Model (Hz)':<15}{'Amp.':<10}{'Valós (Hz)':<15}{'Amp.':<10}\n")
        f.write("-" * 60 + "\n")
        for i in range(10):
            sim_freq, sim_amp = top_sim[i]
            real_freq, real_amp = top_real[i]
            f.write(f"f{i+1:<4} {sim_freq:<15.2f}{sim_amp:<10.2f}{real_freq:<15.2f}{real_amp:<10.2f}\n")
    print(f" TXT export kész: {filename}")

# csv export
def export_peaks_to_csv(freqs_sim, spec_sim, freqs_real, spec_real, filename="spektrum_osszehasonlitas.csv"):
    peaks_sim, _ = find_peaks(spec_sim, height=0.05)
    peaks_real, _ = find_peaks(spec_real, height=0.05)
    top_sim = sorted(zip(freqs_sim[peaks_sim], spec_sim[peaks_sim]), key=lambda x: -x[1])[:10]
    top_real = sorted(zip(freqs_real[peaks_real], spec_real[peaks_real]), key=lambda x: -x[1])[:10]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Harmonikus", "Elméleti (Hz)", "Model (Hz)", "Valós (Hz)", "Eltérés (Hz)", "Megjegyzés"])
        for i in range(10):
            elmeleti = (i + 1) * f1
            sim_freq, _ = top_sim[i]
            real_freq, _ = top_real[i]
            delta = abs(sim_freq - real_freq)
            note = " Egybeesik" if delta < 1 else " Kis eltérés" if delta < 3 else " Jelentős"
            writer.writerow([f"f{i+1}", f"{elmeleti:.2f}", f"{sim_freq:.2f}", f"{real_freq:.2f}", f"{delta:.2f}", note])
    print(f" CSV export kész: {filename}")

# saving
save_top_peaks_txt(freq_sim, spec_sim, freq_real, spec_real)
export_peaks_to_csv(freq_sim, spec_sim, freq_real, spec_real)

print("\n fullos")
