import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_theme(style="whitegrid", context="notebook")

fig_size = (14, 11)
font_size = 28
line_width = 9

plt.rcParams.update({
    "figure.figsize": fig_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "lines.linewidth": line_width,
})


def plot_spectrogram(t, f, zxx, time_unit="s", shading="gouraud"):

    if time_unit == "min":
        t = t / 60.0
    fig, ax = plt.subplots(figsize=fig_size)
    mesh = ax.pcolormesh(t, f, zxx, shading=shading, cmap="plasma")
    cbar = fig.colorbar(mesh, ax=ax, label="dB")
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("dB", size=font_size)
    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel("Frequency [Hz]")
    fig.tight_layout()
    plt.show()


wav_file = "coastal studies institute audio ncroep hat01 cf1e audio CF1E 20150122 070000Z.wav"

rate, data = wavfile.read(wav_file)
data = data.astype(np.float64)

f, t, zxx = sps.stft(data, fs=rate, nperseg=2**11, noverlap=2**10)
zxx_db = 20 * np.log10(np.abs(zxx) + 1e-6)

plot_spectrogram(t, f, zxx_db, time_unit="min", shading="auto")


X_input = np.load("INPUT.npy", allow_pickle=True)


t, f, zxx = X_input[553]        
plot_spectrogram(t, f, zxx, time_unit="s") 
