import crepe
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np


def draw_fundamental_curve(time_arr, freq_arr):
    fig = plt.figure()
    # percentage of axes occupied
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(time_arr, freq_arr, ls=':', alpha=1, lw=1, zorder=1)
    axes.set_title('Pitch Curve')

    plt.show()


if __name__ == '__main__':
    sr, audio = wavfile.read('background.wav')
    # viterbi: smoothing for the pitch curve
    # step_size: 10 milliseconds
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=False, step_size=10)
    draw_fundamental_curve(time, frequency)
