import json
import math
import os.path
import crepe
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import freq_position
import librosa
from triangulation.smooth import Savgol_Filter
from triangulation.triangulation import visualize_3d


def draw_fundamental_curve(time_arr, freq_arr, conf_arr, proj, algo):
    fig = plt.figure(figsize=(10, 8))
    # percentage of axes occupied
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.8])
    # axes.plot(time_arr, freq_arr, ls=':', alpha=1, lw=1, zorder=1)
    ax = axes.scatter(time_arr, freq_arr, c=conf_arr, s=1.5, cmap="OrRd")
    axes.set_title('Pitch Curve')
    fig.colorbar(ax)
    # plt.show()
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists(f'output/{proj}'):
        os.mkdir(f'output/{proj}')
    plt.savefig(f'output/{proj}/pitch_curve_{algo}.jpg')


def draw_contact_points(data, proj):
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

    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists(f'output/{proj}'):
        os.mkdir(f'output{proj}')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{proj}/virtual_contact_point.avi', fourcc, fps=30, frameSize=[700, 1000])
    for frame_idx, frame in enumerate(data):
        print(f'Frame {frame_idx + 1}...')
        points = []
        for idx, ratio in enumerate(frame):
            if ratio >= 0:
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
            print(f'No points detected in Frame {frame_idx + 1}')

        # Add a title and legend
        ax.set_title(f'Cello Strings (Frame {frame_idx + 1})')
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


def pitch_detect_crepe(proj, center, audio_path='wavs/background.wav'):
    sr, audio = wavfile.read(audio_path)
    # viterbi: smoothing for the pitch curve
    # step_size: 10 milliseconds
    # center: False, don't need to pad!
    sample_num = audio.shape[0]
    ic(audio.shape)
    ic(sr)
    frame_num = math.floor(sample_num / sr * 30)
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True, step_size=100 / 3, model_capacity='full', center=center)
    draw_fundamental_curve(time, frequency, confidence, proj, 'crepe')
    pitch_results = np.stack((time, frequency, confidence), axis=1)
    # Pitch Data Persistence
    # np.savetxt("pitch.csv", pitch_results, delimiter=",")
    ic(pitch_results.shape)
    pitch_results = pitch_results[:frame_num, :]
    # ic(pitch_results)
    return pitch_results


def pitch_detect_pyin(proj, audio_path='wavs/background.wav'):
    # sr, audio = wavfile.read(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    # viterbi: smoothing for the pitch curve
    # step_size: 10 milliseconds
    # center: False, don't need to pad!
    sample_num = y.shape[0]
    ic(y.shape)
    ic(sr)
    frame_num = math.floor(sample_num / sr * 30)
    ic(frame_num)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, frame_length=1600, hop_length=1600, center=True,
                                                 fmin=librosa.note_to_hz('B1'), fmax=librosa.note_to_hz('C6'))
    time = librosa.times_like(f0)
    draw_fundamental_curve(time, f0, voiced_probs, proj, 'pyin')
    pitch_results = np.stack((time, f0, voiced_probs), axis=1)
    # Pitch Data Persistence
    # np.savetxt("pitch.csv", pitch_results, delimiter=",")
    ic(pitch_results.shape)
    pitch_results = pitch_results[:frame_num, :]
    ic(pitch_results)
    return pitch_results


def cal_dist(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))


def mapping(proj_dir, positions):
    """
    positions: n * 4
    """
    with open(f'../kp_3d_result/{proj_dir}/kp_3d_all.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_all'])
    ic(kp_3d_all.shape)

    # positions = np.ones([712, 4]) * -1  # n, 4
    # positions[0] = np.array([-1, 0, 1 / 2, -1])
    # positions[1] = np.array([-1, 1 / 2, 1 / 3, -1])
    kp_3d_all_with_cp = kp_3d_all.copy().tolist()
    last_freq = np.nan
    vibrato_freq_list = []
    used_finger_index = np.nan
    used_finger = []

    for frame, kp_3d in enumerate(kp_3d_all):
        # ic(kp_3d.shape)
        finger_board = []
        string_4_top = kp_3d[134, :]
        string_1_top = kp_3d[135, :]
        string_4_bottom = kp_3d[136, :]
        string_1_bottom = kp_3d[137, :]
        string_3_top = string_4_top + (string_1_top - string_4_top) * 1 / 3
        string_3_bottom = string_4_bottom + (string_1_bottom - string_4_bottom) * 1 / 3
        string_2_top = string_4_top + (string_1_top - string_4_top) * 2 / 3
        string_2_bottom = string_4_bottom + (string_1_bottom - string_4_bottom) * 2 / 3
        finger_board.append(string_1_top - string_1_bottom)
        finger_board.append(string_2_top - string_2_bottom)
        finger_board.append(string_3_top - string_3_bottom)
        finger_board.append(string_4_top - string_4_bottom)

        position = positions[frame]
        # ic(position)
        # Used to filter out the contact point closest to the playing wrist
        # TODO： if ratio == 0, how to calculate the distance?
        left_wrist = kp_3d[91, :]

        index_mcp = kp_3d[96, :]
        middle_mcp = kp_3d[100, :]
        ring_mcp = kp_3d[104, :]
        pinky_mcp = kp_3d[108, :]

        index_pip = kp_3d[97, :]
        middle_pip = kp_3d[101, :]
        ring_pip = kp_3d[105, :]
        pinky_pip = kp_3d[109, :]

        index_dip = kp_3d[98, :]
        middle_dip = kp_3d[102, :]
        ring_dip = kp_3d[106, :]
        pinky_dip = kp_3d[110, :]

        index_tip = kp_3d[99, :]
        middle_tip = kp_3d[103, :]
        ring_tip = kp_3d[107, :]
        pinky_tip = kp_3d[111, :]

        rf_middle = middle_mcp + (middle_pip - middle_mcp) * 0.5
        rf_ring = ring_mcp + (ring_pip - ring_mcp) * 0.5
        rf = rf_middle + (rf_ring - rf_middle) * 0.5

        mcps = [index_mcp, middle_mcp, ring_mcp, pinky_mcp]
        pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        dips = [index_dip, middle_dip, ring_dip, pinky_dip]
        tips = [index_tip, middle_tip, ring_tip, pinky_tip]

        contact_point = [np.nan, np.nan, np.nan]
        dist = np.inf
        current_freq = np.nan

        contact_point_list = []
        dist_list = []
        current_freq_list = []

        for pos_idx, ratio in enumerate(position):
            if ratio > 0:
                # temp_contact_point = finger_board[pos_idx] * ratio + locals()[f'string_{pos_idx + 1}_bottom']
                # temp_dist = cal_dist(rf, temp_contact_point)
                # if temp_dist < dist:
                #     dist = temp_dist
                #     contact_point = temp_contact_point
                #     string_fund_freq = freq_position.PITCH_RANGES[pos_idx][0]
                #     current_freq = freq_position.positon2freq(string_fund_freq, ratio)
                potential_contact_point = finger_board[pos_idx] * ratio + locals()[f'string_{pos_idx + 1}_bottom']
                potential_dist = cal_dist(rf, potential_contact_point)
                dist_list.append(potential_dist)
                contact_point_list.append(potential_contact_point)
                string_fund_freq = freq_position.PITCH_RANGES[pos_idx][0]
                current_freq_list.append(freq_position.positon2freq(string_fund_freq, ratio))
        
        smallest_dist = np.inf
        for i in np.argsort(dist_list)[:2]:  # Obtain two closest potential contact points
            for tip in dips:
                temp_dist = cal_dist(tip, contact_point_list[i])
                if temp_dist < smallest_dist:
                    smallest_dist = temp_dist
                    dist = dist_list[i]
                    contact_point = contact_point_list[i]
                    current_freq = current_freq_list[i]
        
        used_finger_mcp = [np.nan, np.nan, np.nan]
        used_finger_pip = [np.nan, np.nan, np.nan]
        used_finger_dip = [np.nan, np.nan, np.nan]
        used_finger_tip = [np.nan, np.nan, np.nan]
        within_mean_range = False
        if not np.isnan(current_freq):
            within_last_range = freq_position.cent_dev(last_freq, -30) < current_freq < freq_position.cent_dev(last_freq, 30)
            if within_last_range:
                # Potential Vibrato Detected
                vibrato_freq_list.append(current_freq)
                vibrato_freq_mean = np.mean(vibrato_freq_list)
                within_mean_range = freq_position.cent_dev(vibrato_freq_mean, -30) < current_freq < freq_position.cent_dev(vibrato_freq_mean, 30)
                if within_mean_range:
                    # Vibrato Detected and Finger Not Changed
                    used_finger_mcp = mcps[used_finger_index]
                    used_finger_pip = pips[used_finger_index]
                    used_finger_dip = dips[used_finger_index]
                    used_finger_tip = tips[used_finger_index]
            elif not within_last_range or (within_last_range and not within_mean_range):
                # Finger Changed
                dist_tip_cp = np.inf
                for finger_id, tip in enumerate(tips):
                    temp_dist_tip_cp = cal_dist(tip, contact_point)
                    # if frame in [i for i in range(352, 360)]:
                    #     print(frame, finger_id, temp_dist_tip_cp)
                    if temp_dist_tip_cp < dist_tip_cp:
                        dist_tip_cp = temp_dist_tip_cp
                        used_finger_mcp = mcps[finger_id]
                        used_finger_pip = pips[finger_id]
                        used_finger_dip = dips[finger_id]
                        used_finger_tip = tips[finger_id]
                        used_finger_index = finger_id
                print(f'Frame {frame} change to finger {used_finger_index + 1}.')
                vibrato_freq_list = []
            last_freq = current_freq

        # ic(contact_point)
        temp_list = kp_3d_all_with_cp[frame]
        # index from 142 to 154
        temp_list.extend(
            [list(string_4_top), list(string_4_bottom), list(string_3_top), list(string_3_bottom), list(string_2_top),
             list(string_2_bottom), list(string_1_top), list(string_1_bottom), list(contact_point),
             list(used_finger_mcp), list(used_finger_pip), list(used_finger_dip), list(used_finger_tip)])
        used_finger.append(used_finger_index)
        kp_3d_all_with_cp[frame] = temp_list

    kp_3d_all_with_cp = np.array(kp_3d_all_with_cp)
    ic(kp_3d_all_with_cp.shape)
    # Bow Points are currently not available, otherwise index should be set to 142
    kp_3d_all = kp_3d_all_with_cp[:, :140, :]
    kp_3d_all_smooth = Savgol_Filter(kp_3d_all, 140)
    ic(kp_3d_all_smooth.shape)
    cp = kp_3d_all_with_cp[:, 140:, :]
    kp_3d_all_with_cp_smooth = np.concatenate((kp_3d_all_smooth, cp), axis=1)
    for frame_num, finger in enumerate(used_finger):
        if True not in np.isnan(kp_3d_all_with_cp[frame_num][150]):  # whether contact point (index: 150) exists
            for i in range(96, 100):
                kp_3d_all_with_cp_smooth[frame_num][finger + i] = kp_3d_all_with_cp[frame_num][finger + i]
    ic(kp_3d_all_with_cp_smooth.shape)
    visualize_3d(kp_3d_all_with_cp_smooth, proj_dir, 'cp_smooth_3d')
    data_dict = {'kp_3d_all_with_cp_smooth': kp_3d_all_with_cp_smooth.tolist()}
    with open(f'kp_3d_all_with_cp_smooth.json', 'w') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    proj_dir = 'cello_1113_scale'
    pitch_results = pitch_detect_crepe(proj_dir, True, 'wavs/scale.wav')
    # ic(pitch_results[352:360])
    # pitch_results = pitch_detect_pyin(proj_dir, 'wavs/scale.wav')
    pitch_with_positions = freq_position.get_contact_position(pitch_results)
    ic(pitch_with_positions.shape)
    positions = pitch_with_positions[:, -4:]
    ic(positions.shape)
    # draw_contact_points(positions, proj_dir)
    mapping(proj_dir, positions)
