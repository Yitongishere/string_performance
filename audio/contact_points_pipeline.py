import argparse
import json
import math
import os.path
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import freq_position
import librosa
import sys
sys.path.append('..')
from triangulation.smooth import Savgol_Filter
from triangulation.triangulation_pipeline import visualize_3d
from triangulation.triangulation_pipeline import FULL_FINGER_INDICES


def draw_fundamental_curve(time_arr, freq_arr, conf_arr, proj, algo):
    fig = plt.figure(figsize=(10, 8))
    # percentage of axes occupied
    axes = fig.add_axes([0.1, 0.1, 0.9, 0.8])
    # axes.plot(time_arr, freq_arr, ls=':', alpha=1, lw=1, zorder=1)
    ax = axes.scatter(time_arr, freq_arr, c=conf_arr, s=1.5, cmap="OrRd")
    axes.set_title('Pitch Curve')
    fig.colorbar(ax)
    # plt.show()
    
    if not os.path.exists(f'output/{proj}'):
        os.makedirs(f'output/{proj}', exist_ok=True)
    plt.savefig(f'output/{proj}/pitch_curve_{algo}.jpg')


def draw_contact_points(data, file_name, proj):
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
    
    if not os.path.exists(f'output/{proj}'):
        os.makedirs(f'output/{proj}', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'output/{proj}/{file_name}.avi', fourcc, fps=30, frameSize=[700, 1000])
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


def pitch_detect_crepe(crepe_backend, proj, instrument='cello', audio_path='wavs/background.wav'):

    # viterbi: smoothing for the pitch curve
    # step_size: 10 milliseconds
    # center: False, don't need to pad!
    
    if crepe_backend == 'torch':
        import torchcrepe
        import torch
        #audio, sr = torchcrepe.load.audio(audio_path)
        audio, sr = librosa.load(audio_path, mono=True)
        audio_1channel = torch.tensor(audio).reshape(1,-1)
        sample_num = audio_1channel.shape[1]
        if instrument == 'cello':
            freq_range = freq_position.PITCH_RANGES_CELLO
        else:
            freq_range = freq_position.PITCH_RANGES_VIOLIN
        min_freq = np.min(freq_range)
        max_freq = np.max(freq_range)
        frame_num = math.floor(30 * sample_num / sr)
        frequency,confidence = torchcrepe.predict(audio_1channel,
                                                   sr,
                                                   hop_length=int(sr / 30.),
                                                   return_periodicity=True,
                                                   model='full',
                                                   fmin = min_freq,
                                                   fmax = max_freq,
                                                   batch_size=2048,
                                                   device='cuda:0')
        
        
        frequency = frequency.reshape(-1,)
        confidence = confidence.reshape(-1,)
        time = np.arange(0,100/3.*frequency.size()[0],100/3.).reshape(-1,)[:len(frequency)]
        print(frequency)
    elif crepe_backend == 'tensorflow':
        import crepe
        sr, audio = wavfile.read(audio_path)
        sample_num = audio.shape[0]
        frame_num = math.floor(30 * sample_num / sr)
        time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True, step_size=100 / 3, model_capacity='full', center=True)
        print(frequency)
    else:
        print('the argument "crepe_backend" is either "tensorflow" or "torch"')
    
    draw_fundamental_curve(time, frequency, confidence, proj, 'crepe')
    
    pitch_results = np.stack((time, frequency, confidence), axis=1)
    # Pitch Data Persistence
    # np.savetxt("pitch.csv", pitch_results, delimiter=",")
    ic(pitch_results.shape)
    pitch_results = pitch_results[:frame_num, :]
    print(frame_num)
    print(pitch_results.shape)
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


def point_init():
    return np.array([np.nan, np.nan, np.nan])


def mapping(proj, positions, instrument = 'cello', visualize=False):
    """
    positions: n * 4
    """
    with open(f'../triangulation/kp_3d_result/{proj}/kp_3d_all_dw.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_all_dw'])
    ic(kp_3d_all.shape)

    # with open(f'../pose_estimation/{proj_dir}/kp_3d_all_pe.json', 'r') as f:
    #     data_dict = json.load(f)
    # kp_3d_all = np.array(data_dict['kp_3d_all_pe'])
    # ic(kp_3d_all.shape)

    # positions = np.ones([712, 4]) * -1  # n, 4
    # positions[0] = np.array([-1, 0, 1 / 2, -1])
    # positions[1] = np.array([-1, 1 / 2, 1 / 3, -1])
    kp_3d_all_cp = kp_3d_all.copy().tolist()
    last_freq = np.nan
    vibrato_freq_list = []
    used_finger_index = np.nan
    used_finger = []
    filtered_positions = []
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

        contact_point = point_init()
        dist = np.inf
        current_freq = np.nan

        contact_point_list = []
        dist_list = []
        current_freq_list = []
        pressed_string_id_list = []
        
        if instrument == 'cello':
            freq_range = freq_position.PITCH_RANGES_CELLO
        else:
            freq_range = freq_position.PITCH_RANGES_VIOLIN
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
                
                string_fund_freq = freq_range[pos_idx][0]
                current_freq_list.append(freq_position.positon2freq(string_fund_freq, ratio))
                pressed_string_id_list.append(pos_idx)

        smallest_dist = np.inf
        pressed_string_id = -1
        for i in np.argsort(dist_list)[:2]:  # Obtain two closest potential contact points
            for tip in tips:
                temp_dist = cal_dist(tip, contact_point_list[i])
                if temp_dist < smallest_dist:
                    smallest_dist = temp_dist
                    dist = dist_list[i]
                    contact_point = contact_point_list[i]
                    current_freq = current_freq_list[i]
                    pressed_string_id = pressed_string_id_list[i]
                    # ic(pressed_string_id)
                    # ic(i)
        
        used_finger_mcp = point_init()
        used_finger_pip = point_init()
        used_finger_dip = point_init()
        used_finger_tip = point_init()
        within_mean_range = False
        if not np.isnan(current_freq):
            within_last_range = freq_position.cent_dev(last_freq, -60) < current_freq < freq_position.cent_dev(last_freq, 60)
            if within_last_range:
                # Potential Vibrato Detected
                vibrato_freq_list.append(current_freq)
                vibrato_freq_mean = np.mean(vibrato_freq_list)
                within_mean_range = freq_position.cent_dev(vibrato_freq_mean, -60) < current_freq < freq_position.cent_dev(vibrato_freq_mean, 60)
                if within_mean_range:
                    # Vibrato Detected and Finger Not Changed
                    used_finger_mcp = mcps[used_finger_index]
                    used_finger_pip = pips[used_finger_index]
                    used_finger_dip = dips[used_finger_index]
                    used_finger_tip = tips[used_finger_index]
                    # finger not change! update contact point...
                    smallest_dist = np.inf
                    for idx, cp in enumerate(contact_point_list):
                        temp_dist = cal_dist(used_finger_tip, cp)
                        if temp_dist < smallest_dist:
                            smallest_dist = temp_dist
                            contact_point = contact_point_list[idx]
                            pressed_string_id = pressed_string_id_list[idx]
                
            if (not within_last_range) or (within_last_range and not within_mean_range):
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

        # The distance between contact point and fingertip should be limited.
        # The cello body still vibrates even after the finger no longer presses the string.
        if not np.isnan(contact_point).any():
            dist_tip_cp = cal_dist(contact_point, used_finger_tip)
            # threshold that finger is lifted
            dist_tip_pip = cal_dist(used_finger_tip, used_finger_pip)
            if dist_tip_cp > dist_tip_pip:  # used finger lifted
                contact_point = point_init()
                used_finger_mcp = point_init()
                used_finger_pip = point_init()
                used_finger_dip = point_init()
                used_finger_tip = point_init()

        filtered_position = [-1, -1, -1, -1]
        if not np.isnan(contact_point).any():
            filtered_position[pressed_string_id] = position[pressed_string_id]
            filtered_positions.append(filtered_position)
        else:
            filtered_positions.append(filtered_position)

        # ic(contact_point)
        temp_list = kp_3d_all_cp[frame]
        # index from 142 to 154
        temp_list.extend(
            [list(string_4_top), list(string_4_bottom), list(string_3_top), list(string_3_bottom), list(string_2_top),
             list(string_2_bottom), list(string_1_top), list(string_1_bottom), list(contact_point),
             list(used_finger_mcp), list(used_finger_pip), list(used_finger_dip), list(used_finger_tip)])
        used_finger.append(used_finger_index)
        kp_3d_all_cp[frame] = temp_list

    kp_3d_all_cp = np.array(kp_3d_all_cp)
    # ic(kp_3d_all_cp.shape)
    # Bow Points are currently not available, otherwise index should be set to 142
    kp_3d_all = kp_3d_all_cp[:, :140, :]
    kp_3d_all_smooth = Savgol_Filter(kp_3d_all, 140)
    cp = kp_3d_all_cp[:, 140:, :]
    kp_3d_all_cp_smooth = np.concatenate((kp_3d_all_smooth, cp), axis=1)
    for frame_id, finger_id in enumerate(used_finger):
        if not np.isnan(finger_id):
           kp_3d_all_cp_smooth[frame_id][151:155] = kp_3d_all_cp_smooth[frame_id][FULL_FINGER_INDICES[finger_id]]

    # for frame_num, finger in enumerate(used_finger):
    #     if True not in np.isnan(kp_3d_all_cp[frame_num][150]):  # whether contact point (index: 150) exists
    #         for i in range(96, 100):
    #             kp_3d_all_cp_smooth[frame_num][finger + i] = kp_3d_all_cp[frame_num][finger + i]
    # ic(kp_3d_all_cp_smooth.shape)
    if visualize:
        visualize_3d(kp_3d_all_cp_smooth, proj_dir, 'dw_cp_smooth_3d', 'whole')

    # Smooth one is the final result of pure dwpose result
    # Origin one will be sent to further processing
    data_dict_smooth = {'kp_3d_all_dw_cp_smooth': kp_3d_all_cp_smooth.tolist()}
    data_dict_origin = {'kp_3d_all_dw_cp': kp_3d_all_cp.tolist()}
    
    if not os.path.exists(f'cp_result/{proj}'):
        os.makedirs(f'cp_result/{proj}', exist_ok=True)
    
    with open(f'cp_result/{proj}/kp_3d_all_dw_cp_smooth.json', 'w') as f:
        json.dump(data_dict_smooth, f)
    with open(f'cp_result/{proj}/kp_3d_all_dw_cp.json', 'w') as f:
        json.dump(data_dict_origin, f)
    return filtered_positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='contact_points_pipeline')
    parser.add_argument('--wav_path', default='wavs/scale_128_786.wav', type=str, required=True)
    parser.add_argument('--parent_dir', default='cello', type=str, required=True)
    parser.add_argument('--proj_dir', default='cello01', type=str, required=True)
    parser.add_argument('--instrument', default='cello', type=str, required=True)
    parser.add_argument('--visualize', default=False, required=False, action='store_true')
    parser.add_argument('--draw_cps', default=False, required=False, action='store_true')
    parser.add_argument('--draw_filtered_cps', default=False, required=False, action='store_true')
    parser.add_argument('--save_position', default=False, required=False, action='store_true')
    parser.add_argument('--crepe_backend', default='torch', required=False)
    args = parser.parse_args()
    
    wav_path = args.wav_path
    parent_dir = args.parent_dir
    proj_dir = args.proj_dir
    instrument = args.instrument
    visualize = args.visualize
    draw_cps = args.draw_cps
    draw_filtered_cps = args.draw_filtered_cps
    save_position = args.save_position
    crepe_backend = args.crepe_backend
    
    proj = parent_dir + os.sep + proj_dir
    
    pitch_results = pitch_detect_crepe(crepe_backend, proj, instrument, wav_path)
    print(pitch_results)
    print(pitch_results.shape)
    pitch_with_positions = freq_position.get_contact_position(pitch_results,instrument)
    positions = pitch_with_positions[:, -4:]
    print(positions)
    if draw_cps:
        draw_contact_points(positions, proj, 'virtual_contact_point')
    new_positions = mapping(proj, positions, visualize=visualize)
    if save_position:
        np.savetxt(f'positions_{proj_dir}', new_positions)
    if draw_filtered_cps:
        draw_contact_points(new_positions, proj, 'virtual_contact_point_filtered')
