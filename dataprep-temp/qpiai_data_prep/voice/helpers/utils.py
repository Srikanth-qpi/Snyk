import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class voice_ops:
    def __init__(self, x, sample_rate):
        self.x = x
        self.sample_rate = sample_rate

    def waveform(x, sample_rate, filename):
        librosa.display.waveplot(x, sample_rate)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        return plt.savefig(filename)

    def denoise(x, sample_rate, filename):
        chroma = librosa.feature.chroma_cqt(x, sample_rate)
        chroma_med = librosa.decompose.nn_filter(
            chroma, aggregate=np.median, metric="cosine"
        )
        plt.figure(figsize=(20, 18))
        plt.subplot(5, 1, 2)
        librosa.display.specshow(chroma_med, y_axis="chroma")
        plt.colorbar()
        plt.title("Median-filtered")
        return plt.savefig(filename)

    def mfcc(x, sample_rate, filename):
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature = mfccs
        output_df = pd.DataFrame(feature, columns=["output"])
        return output_df.to_csv(filename)

    def fft(x, sample_rate, filename):
        fft = np.fft.fft(x)
        magnitude = np.abs(fft)
        frequency = np.linspace(0, sample_rate, len(magnitude))
        plt.plot(frequency, magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        return plt.savefig(filename)

    def sft(x, sample_rate, filename):
        plt.figure(figsize=(10, 10))
        stft = librosa.feature.melspectrogram(x, sample_rate, n_mels=128, fmax=8000)
        librosa.display.specshow(
            librosa.power_to_db(stft, ref=np.max), y_axis="mel", fmax=8000
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel spectrogram")
        plt.tight_layout()
        return plt.savefig(filename)

    def beats_count(x, sample_rate, filename):
        tempo, frames = librosa.beat.beat_track(x, sample_rate)
        my_dict = {
            "frame": frames,
            "second": librosa.frames_to_time(frames, sample_rate),
        }
        df = pd.DataFrame(my_dict)
        return df.to_csv(filename, header=False, index=False)

    def noise_addition(x, sample_rate, filename):
        wav_n = x + 0.009 * np.random.normal(0, 1, len(x))
        return librosa.output.write_wav(filename, wav_n, sample_rate)

    def shift_time(x, sample_rate, filename):
        wav_roll = np.roll(x, int(sample_rate / 10))
        return librosa.output.write_wav(filename, wav_roll, sample_rate)

    def stretch_time(x, sample_rate, filename):
        factor = 0.4
        wav_time_stch = librosa.effects.time_stretch(x, factor)
        return librosa.output.write_wav(filename, wav_time_stch, sample_rate)

    def shift_pitch(x, sample_rate, filename):
        wav_pitch_sf = librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=-5)
        return librosa.output.write_wav(filename, wav_pitch_sf, sample_rate)
