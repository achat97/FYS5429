import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
from scipy.stats import zscore
from pathlib import Path



def read_resample(file, name, fs, read_only=False):

    """
    Reads a .wav file and resamples it by default to a new sample rate fs.
    The (resampled) data is saved as a .npy file.
    
    ----------

    Parameters:
        file (str or pathlib.Path) - filename or path to .wav file.
        name (str) - name of the output datafile.
        fs (int) - target sampling frequency in Hz.
        read_only (bool) - default set to False. If True, the file will be saved as a .npy without a change in sampling frequency.

    Returns:
        data (ndarray) - 1-D array containing the (resampled) time-series values.
    """

    rate, data = wavfile.read(file)

    if read_only:
        np.save(name + ".npy", data)
        print(rate)
        return data

    new_rate = fs
    nsamples = round(len(data) * new_rate / rate)
    data_resampled = sps.resample(data, nsamples)
    np.save(name + ".npy", data_resampled)

    print("Raw data")
    print("------------------------------")
    print(f"Number of data points: {len(data)}")
    print(f"Audio duration: {round(1/60 * len(data)/rate)} minutes \n")

    print("Resampled data")
    print("------------------------------")
    print(f"Number of data points: {len(data_resampled)}")
    print(f"Audio duration: {round(1/60 * len(data_resampled)/rate)} minutes \n")

    return data_resampled


def split(data,fs,size,name):

    """
    Divides a time-series into segments of equal length and returns both the divided signal and the corresponding time stamps separately. 
    The segmented signal is saved as a .npy file.

    ----------

    Parameters:
        data (ndarray) - 1-D array containing the time-series values.
        fs (int) - sampling frequency of the data.
        size (int) - length of each segment, expressed in seconds. 
        name (str) - name of the data segments file. 

    Returns:
        data_segments (ndarray) - 2-D array of the time-series where each row is a segment of the original signal,
                                  with shape (number of segments, samples per segment).
        t_segments (ndarray) - 2-D array containing the time values (in seconds) corresponding to each sample
                               in the respective segment, with shape (number of segments, samples per segment).
    """
    
    chunk_size = size * fs
    n_chunks = len(data) // chunk_size

    N = n_chunks * chunk_size
    data_trimmed = data[:N]

    data_segments = data_trimmed.reshape(n_chunks ,chunk_size)

    t = np.linspace(0,(N-1)/fs,N)
    
    t_segments = t.reshape(n_chunks ,chunk_size)

    np.save(name+"_segments"+".npy",data_segments)

    return data_segments, t_segments
    

def noise_normalize(data,name):

    """
    Normalizes each segment of a 2-D time-series array independently using z-score normalization and returns the normalized segments. 
    The normalized data is saved as a .npy file.

    ----------

    Parameters:
        data (ndarray) - 2-D array of time-series segments with shape (number of segments, samples per segment).
        name (str) - name of the normalized datafile.

    Returns:
        data_normalized (ndarray) - 2-D array of the z-score normalized time-series segments,
                                    with shape (number of segments, samples per segment).                     
    """
    
    data_normalized = []

    for i in range(data.shape[0]):
        data_normalized.append(zscore(data[i]))

    data_normalized = np.array(data_normalized)
    np.save(name+"_normalized"+".npy",data_normalized)

    return data_normalized


def spectrograms(data,fs,name):

    """
    Computes the spectrogram of each segment in a 2-D time-series array using the Short-Time Fourier Transform (STFT)
    and returns the resulting spectrograms. A segment length of 2048 samples (2**11) with 50% overlap (2**10) is used.
    The spectrograms are saved as a .npy file.
    
    ----------

    Parameters:
        data (ndarray) - 2-D array of time-series segments with shape (number of segments, samples per segment).
        fs (int) - sampling frequency of the data in Hz.
        name (str) - name of the spectrograms datafile.

    Returns:
    spectrogram (ndarray) - 3-D array containing the magnitude spectrogram of each segment,
                            with shape (number of segments, frequency bins, time bins).                       
    """

    spectrogram = []

    for i in range(data.shape[0]):
        f,t,zxx = sps.stft(data[i], fs=fs, nperseg=2**11, noverlap=2**10)
        spectrogram.append(np.abs(zxx))

        print(f"Spectrogram {i+1}/{data.shape[0]}", end="\r")
    
    print("")
         
    spectrogram = np.array(spectrogram)
    np.save(name+"_spectrograms"+".npy",spectrogram)

    return spectrogram


file_name = "ocean_noise"
fs_new = 30000
seg_length = 5

#data_resampled = read_resample("coastal studies institute audio ncroep hat01 cf1e audio CF1E 20150122 070000Z.wav",file_name,fs_new)
#data_segments, t_segments = split(data_resampled,fs_new,seg_length,file_name)
#data_segments_normalized = noise_normalize(data_segments,file_name)


