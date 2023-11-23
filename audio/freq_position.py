import numpy as np
from icecream import ic

# 标准音高
BENCHMARK_A0 = 440 / np.power(2, 4)
# ic(BENCHMARK_A0)

# 12平均率相邻半音间的倍数关系
R = np.power(2, 1 / 12)
# ic(R)

# 计算得到全音域每个半音的频率（midi 0 - 127）
NOTE_FREQ = []
# A0 在 midi中对应 21
for i in range(0, 22):
    ith_note = BENCHMARK_A0 / np.power(R, i)
    NOTE_FREQ.insert(0, ith_note)

for i in range(1, 128 - 21):
    ith_note = BENCHMARK_A0 * np.power(R, i)
    NOTE_FREQ.append(ith_note)
# ic(NOTE_FREQ)

# 定义每根弦的音域
# PITCH_RANGES = [
#     [NOTE_FREQ[57], NOTE_FREQ[84]],  # 一弦  A3: 220.000 Hz --  C6: 1046.502 Hz
#     [NOTE_FREQ[50], NOTE_FREQ[62]],  # 二弦  D3: 146.832 Hz --  D4: 293.665 Hz
#     [NOTE_FREQ[43], NOTE_FREQ[55]],  # 三弦  G2: 97.999 Hz  --  G3: 195.998 Hz
#     [NOTE_FREQ[36], NOTE_FREQ[48]],  # 四弦  C2: 65.406 Hz  --  C3: 130.813 Hz
# ]
PITCH_RANGES = [
    [NOTE_FREQ[57], NOTE_FREQ[84]],  # 一弦  A3: 220.000 Hz --  C6: 1046.502 Hz
    [NOTE_FREQ[50], NOTE_FREQ[74]],  # 二弦  D3: 146.832 Hz --  D5: 587.330 Hz
    [NOTE_FREQ[43], NOTE_FREQ[67]],  # 三弦  G2: 97.999 Hz  --  G4: 391.995 Hz
    [NOTE_FREQ[36], NOTE_FREQ[60]],  # 四弦  C2: 65.406 Hz  --  C4: 261.626 Hz
]
# ic(PITCH_RANGES)


def freq2position(fund_freq, cur_freq):
    """
    On a target string with fund_freq as fundamental frequency, for a specific frequency cur_freq,
    infer the effective finger position (the vibration length of the string)

    Parameters:
      fund_freq - fundamental frequency of the target string
      cur_freq - specific frequency to be converted

    Returns:
        effective finger position (taking the bottom side as 0, while top side as 1
    """
    position = fund_freq / cur_freq
    return position


def positon2freq(fund_freq, position):
    cur_freq = fund_freq / position
    return cur_freq


def cent2freq(cent_num, ncent_per_note):
    """
    从 midi 音符 0 起，将每个半音程按照音分等分为 ncent_per_note 段，计算第 cent_num 段对应的频率。
    """
    freq = 440 * np.power(2, ((cent_num - (69 * ncent_per_note)) / (12 * ncent_per_note)))
    return freq


def cent_dev(base_note, cent_num):
    """
    基于某个音高，向上偏移 cent_num 个音分后的频率
    """
    freq = base_note * np.power(2, cent_num / 1200)
    return freq


def _write_raw_index(path, text):
    """在csv文件中第一行添加索引字段"""
    with open(path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(text + '\n' + content)


def get_contact_position(pitch_info):
    # 设置音高检测置信度门槛
    conf_threshold = 0.8
    # 空弦判断门槛
    base_threshold = 0.04
    # shape in (n_timesteps, 7)
    #      ->  dim1: (timestep, frequency, confidence, string1_pos, string2_pos, string3_pos, string4_pos)
    info_all = []
    # 对当前时间节点下的音高做处理
    for step in pitch_info:
        position_cur = [step[0], step[1], step[2], -1, -1, -1, -1]
        possible_strings = []
        if step[2] >= conf_threshold:
            freq_cur = step[1]
            # 确定在哪些弦上可得到该音
            for i in range(len(PITCH_RANGES)):
                # 留50音分的误差包容度
                if cent_dev(PITCH_RANGES[i][0], -50) < freq_cur < cent_dev(PITCH_RANGES[i][1], 50):
                    possible_strings.append(i + 1)
            # 确定在各弦上的演奏位置
            if len(possible_strings) != 0:
                for i in possible_strings:
                    position_cur[i + 2] = freq2position(PITCH_RANGES[i - 1][0], freq_cur)
                    if abs(position_cur[i + 2] - 1) < base_threshold:
                        position_cur[i + 2] = 1
        info_all.append(position_cur)
    info_all = np.array(info_all)
    # Pitch and Position Data Persistence
    # np.savetxt("info_all.csv", info_all, delimiter=",")
    # index_text = 'time, frequency, confidence, string1_pos, string2_pos, string3_pos, string4_pos'
    # _write_raw_index(path="info_all.csv", text=index_text)
    return info_all

ic(cent_dev(162.417, 30))
