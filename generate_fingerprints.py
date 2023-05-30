import numpy as np
from scipy.signal import savgol_filter, stft
import matplotlib.pyplot as plt

n_samples = 1000
genuine_signals = [np.sin(2*np.pi*np.random.rand(n_samples) + np.random.rand()*np.pi) for _ in range(100)]

spoofed_signals = [np.sin(2*np.pi*(np.random.rand(n_samples) + np.random.rand()*5) + np.random.rand()*np.pi) for _ in range(100)]
i=0
#To filter out the signal
def filter_signal(signal):
    filtered_signal = savgol_filter(signal, window_length=5, polyorder=2)
    target_signal = signal - filtered_signal
    return target_signal

#Extract the features i.e., the fingerprints
def extract_fingerprint(filtered_data):
    window = 'hann'
    nperseg = 32
    noverlap = 16
    f, t, Zxx = stft(filtered_data, window=window, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)
##for signal in genuine_signals:
##    #noisy_signal = signal + np.random.normal(0, 0.1, n_samples)
##    filtered_signal = filter_signal(signal)
##    fingerprint = extract_fingerprint(filtered_signal)
##    plt.imshow(fingerprint)
##    path = "genuine/"+str(i)+".png"
##    plt.savefig(path,bbox_inches='tight', transparent=True, pad_inches=0)
##    i = i+1

for signal in spoofed_signals:
    
    #noisy_signal = signal + np.random.normal(0, 0.1, n_samples)

    #to add noise, we genarate an artifact(square wave) then add to the signal, which shows as noise spike(intrusion)
        # Define the square wave artifact
    artifact_amp = 0.5  # Amplitude of the artifact
    artifact_dur = 0.05  # Duration of the artifact
    artifact_freq = 50  # Frequency of the artifact
    artifact = np.concatenate((np.zeros(int(0.5*(1-artifact_dur)*1000)), 
                                artifact_amp*np.ones(int(artifact_dur*1000)),
                                np.zeros(int(0.5*(1-artifact_dur)*1000))))
    artifact = np.tile(artifact, int(1000/len(artifact)))

    # Add the artifact to the signal
    signal_with_spike = signal + artifact
    

    
    time_delay = np.random.randint(5, 10)
    #spoofed_signal = np.concatenate([noisy_signal[time_delay:], noisy_signal[:time_delay]])
    spoofed_signal = np.concatenate([signal_with_spike[time_delay:], signal_with_spike[:time_delay]])
    filtered_signal = filter_signal(spoofed_signal)
    fingerprint = extract_fingerprint(filtered_signal)
    plt.imshow(fingerprint)
    path = "spoofed/"+str(i)+".png"
    plt.savefig(path,bbox_inches='tight', transparent=True, pad_inches=0)
    i = i+1
