import crepe
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import freq_position


def draw_fundamental_curve(time_arr, freq_arr, conf_arr):
    fig = plt.figure(figsize=(10, 8))
    # percentage of axes occupied
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.8])
    # axes.plot(time_arr, freq_arr, ls=':', alpha=1, lw=1, zorder=1)
    ax = axes.scatter(time_arr, freq_arr, c=conf_arr, s=1.5, cmap="OrRd")
    axes.set_title('Pitch Curve')
    fig.colorbar(ax)
    plt.show()


def draw_contact_points(data):
    """
    data: [n, 4], n: frame number, 4: string number
    """
    string_low = 1
    string_high = 10
    string_length = string_high - string_low

    # string
    string_xloc = [4, 3, 2, 1]

    # Define the data for four long vertical strings with two points each
    string1 = [(string_xloc[0], string_low), (string_xloc[0], string_high)]
    string2 = [(string_xloc[1], string_low), (string_xloc[1], string_high)]
    string3 = [(string_xloc[2], string_low), (string_xloc[2], string_high)]
    string4 = [(string_xloc[3], string_low), (string_xloc[3], string_high)]

    # Extract x and y coordinates for each string
    x1, y1 = zip(*string1)
    x2, y2 = zip(*string2)
    x3, y3 = zip(*string3)
    x4, y4 = zip(*string4)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'contact_point.avi', fourcc, fps=30, frameSize=[700, 1000])
    for frame_idx, frame in enumerate(data):
        print(f'Frame {frame_idx+1}...')
        points = []
        for idx, ratio in enumerate(frame):
            if ratio > 0:
                points.append([string_xloc[idx], ratio * string_length + string_low])

        # Create a figure and axis without axes
        fig, ax = plt.subplots(figsize=(7, 10))

        ax.set_xlim(0.5, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # matplotlib default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()['color']

        # Plot the four long vertical strings with only two points each
        string1 = ax.plot(x1, y1, marker='o', label='String 1 A', color=colors[0])
        string2 = ax.plot(x2, y2, marker='o', label='String 2 D', color=colors[1])
        string3 = ax.plot(x3, y3, marker='o', label='String 3 G', color=colors[2])
        string4 = ax.plot(x4, y4, marker='o', label='String 4 C', color=colors[4])

        if points:
            for point in points:
                # zorder should be larger than the string number
                ax.scatter(point[0], point[1], color='r', zorder=5)
        else:
            print(f'No points detected in Frame {frame_idx+1}')

        # Add a title and legend
        ax.set_title(f'Cello Strings (Frame {frame_idx+1})')
        num1, num2, num3, num4 = 1.03, 0, 3, 0
        ax.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

        # Swap the aspect ratio of the entire image
        ax.set_aspect(1)

        # Display the plot
        # plt.show()

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(height, width, 3)
        image_array = image_array[:, :, ::-1]  # rgb to bgr
        out.write(image_array)
        plt.close()


def pitch_detect(audio_path='background.wav'):
    sr, audio = wavfile.read(audio_path)
    # viterbi: smoothing for the pitch curve
    # step_size: 10 milliseconds
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True, step_size=33.33, model_capacity='full')
    draw_fundamental_curve(time, frequency, confidence)
    pitch_results = np.stack((time, frequency, confidence), axis=1)
    # Pitch Data Persistence
    # np.savetxt("pitch.csv", pitch_results, delimiter=",")
    return pitch_results


if __name__ == '__main__':
    pitch_results = pitch_detect()
    pitch_with_positions = freq_position.get_contact_position(pitch_results)
    ic(pitch_with_positions.shape)
    positions = pitch_with_positions[:, -4:]
    ic(positions.shape)
    # draw_contact_points(positions)
