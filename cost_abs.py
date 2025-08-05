from sth_abs import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import mean
import efel
from scipy.signal import find_peaks
import math
import warnings
import pickle
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
plt.rcParams['axes.labelsize'] = 16

warnings.filterwarnings("ignore")

def create_updated_cell(f):
    stn_cell = CreateSth(params=f)
    return stn_cell


def get_freq(v, dt):
    peaks = find_peaks(v, height=-30)
    diff_p = np.diff(peaks[0])
    if detect_burst(v):
        mean_freq = 0
    elif len(diff_p) >= 1:
        mean_freq = (1000/dt)/mean(diff_p)
    else:
        mean_freq = 0
    return mean_freq


def detect_burst(v):
    peaks = find_peaks(v, height=-30)
    diff_p = np.diff(peaks[0])
    bursting = False
    for i in range(len(diff_p)):
        if len(diff_p) > 1:  # Should have at least 3 spikes
            if i - 1 < 0:
                i = 1
            if diff_p[i] / diff_p[i - 1] > 1.1:  # This will find the pause between two bursts
                bursting = True
            if diff_p[i - 1] / diff_p[i] > 1.1:
                bursting = True
    return bursting


def get_freq_detect_burst(v, dt):
    freq = get_freq(v, dt)
    bursting = detect_burst(v)
    if bursting:
        freq = 0
    return freq


def find_burst(v, mean_freq, dt):
    peaks = find_peaks(v)
    diff_p = np.diff(peaks[0])
    rest_diff = 20000/(mean_freq + 0.1)
    burst_end = 0
    wait = 0
    for i in range(len(diff_p)):
        if rest_diff/3 < diff_p[i] < rest_diff:
            wait += 1
        elif diff_p[i] >= rest_diff:
            wait -= 1
        if wait > 2:
            burst_end = i
            break
        burst_end = i
    burst_len = peaks[0][burst_end]-peaks[0][0]
    burst_time = burst_len*dt
    return burst_time, peaks[0][0]+int(600/dt)


def cal_score_MHP(v, weight, target):
    peaks = find_peaks(v, height=-20)
    v_MHP = min(v)
    if len(peaks[0]) < 1:
        score_MHP = weight
    elif len(peaks[0]) < 2:
        score_MHP = weight/2
    else:
        d_t = int((peaks[0][1]-peaks[0][0])/8)
        v_check = v[peaks[0][0]:peaks[0][1]]
        peaks_ahp = find_peaks(v_check)
        ahp_pc = len(peaks_ahp[0])
        if ahp_pc > 0:
            v_MHP = min(v_check[peaks_ahp[0][0]:])
        else:
            v_MHP = v[peaks[0][0]+d_t]
        score_MHP = cal_score(v_MHP, target, weight / 2, 'MHP')
    return score_MHP, v_MHP


def cal_score_HP(v, weight, mean_freq, dt, v_peak, v_min):
    spikes = find_peaks(v[int(112.5/dt):int(600/dt)], height=-80)
    if len(spikes[0]) == 0:
        v_HP = min(v[int(112.5/dt):int(600/dt)])
        score_1 = (v_HP - (-80)) / 20 * weight/2
        if score_1 < 0:
            score_1 = 0
        score_shp = 0
        e_slp = v[int(600/dt)] - v_HP
        v_ht = e_slp/2 + v_HP+0.5
        if v[int(350/dt)]>v_ht:
            score_shp = (v[int(350/dt)]-v_ht) / 3 * weight/4
            # print('score_shp: ', score_shp)
        else:
            score_shp = 0
        if e_slp >= 0:
            score_slp = cal_score(e_slp, [4, 10], weight/4, 'SLP')
        else:
            score_slp = weight/4
        score = score_1 + score_slp + score_shp
    else:
        v_HP = 0
        score = weight
        e_slp = 0
    spikes_all = find_peaks(v[int(600/dt):], height=-10)
    burst = detect_burst(v[int(1000/dt):])
    burst_time = 0
    burst_count = 0
    if len(spikes_all[0]) != 0:
        burst_time, first_peak_idx = find_burst(v[int(600/dt):], mean_freq, dt)
        # print(burst_time)
        score_time = cal_score(burst_time, [40, 80], 30, 'burst_time')
        score+=score_time
        end_burst = int(first_peak_idx+burst_time/dt)
        if end_burst > len(v):
            spikes_burst = find_peaks(v[first_peak_idx:], height=-40)
        else:
            spikes_burst = find_peaks(v[first_peak_idx:int(first_peak_idx+burst_time/dt)], height=-40)
        burst_count = len(spikes_burst[0])
        score_bs = cal_score(len(spikes_burst[0]), [4, 10], 30, 'burst_count')
        score+=score_bs
        if burst_count > 0:
            v_bpmin = min(spikes_burst[1]['peak_heights'])
            score_bp_1 = cal_score(v_bpmin, [0, 20], 33, 'burst_peak')
            v_bpmax = max(spikes_burst[1]['peak_heights'])
            score_bp_2 = cal_score(v_bpmax, [0, v_peak], 33, 'burst_peak')
            v_bbmin = min(v[first_peak_idx:])
            score_bp_3 = cal_score(v_bbmin, [v_min, v_min+20], 33, 'burst_peak')
            score_bp = score_bp_1 + score_bp_2 + score_bp_3
            score += score_bp
        else:
            score += 100
            score_bp = 100
        score, max_interval, min_interval, _ = check_dp_block(v, score, 30, mean_freq, dt, threshold=-20)
        score_int = cal_score(min_interval, [3, 10], 30, 'hp_int')
        score+=score_int
    else:
        score+=150
        score_bp = 50
        v_bp = np.mean(v)
        max_interval = burst_time
        min_interval = burst_time
    if burst:
        score += 20
    if score < 0:
        score = 0
    if score > weight:
        score = weight
    if burst == True:
        burst = 1
    else:
        burst = 0
    return score, v_HP, e_slp, v[int(350/dt)], burst, score_bp, burst_time, burst_count, max_interval, min_interval


def check_hp_last_spikes(v, score, weight, dt):
    spikes_after = find_peaks(v[int(750/dt):], height=0)
    if len(spikes_after[0]) != 0:
        score = score
    else:
        score = score + weight
    check_spikes_after = len(spikes_after[0]) != 0
    if check_spikes_after == False:
        check_spikes_after = 0
    else:
        check_spikes_after = 1
    return score, check_spikes_after


def check_dp_block(v, score, weight, mean_freq, dt, threshold=0):
    burst_time, first_peak_idx = find_burst(v[int(600/dt):], mean_freq, dt)
    # print(burst_time)
    end_burst = int(first_peak_idx+burst_time/dt)
    if end_burst > len(v):
        burst_peaks = find_peaks(v[first_peak_idx-100:], height=threshold)
    else:
        burst_peaks = find_peaks(v[first_peak_idx-100:end_burst], height=threshold)
    burst_peaks_time = burst_peaks[0]
    burst_peaks_time = np.append(burst_peaks_time, 100+burst_time/dt)
    burst_interval = np.diff(burst_peaks_time)
    try:
        max_interval = max(burst_interval)*dt
        min_interval = min(burst_interval)*dt
        brr = []
        for i in range(1, len(burst_interval)):
            brr.append(burst_interval[i]/burst_interval[i-1])
        max_brr = max(brr)
    except:
        max_interval = burst_time
        min_interval = burst_time
        max_brr = burst_time
        score_dpb = weight
    if max_brr > 2:
        score_dpb = (max_brr-2)/3 * weight
        score_dpb = np.clip(score_dpb, 0, weight)
    else:
        score_dpb = 0
    score += score_dpb
    score_dur = cal_score(burst_time, [40, 160], weight/2, 'burst_time')
    score += score_dur
    return score, max_brr, min_interval, burst_time


def cal_score(param, target, factor, category):
    if category == 'SP':
        if param < target[0]:
            score = (target[0] - param) / (target[0]/2) * factor
        elif param > target[1]:
            score = (param - target[1]) / (target[1]/2) * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'SLP':
        if param < target[0]:
            score = (target[0] - param) / 4 * factor
        elif param > target[1]:
            score = (param - target[1]) / 4 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'burst_time' or category == 'burst_count' or category == 'hp_int':
        if param < target[0]:
            score = (target[0] - param) / target[0] * factor
        elif param > target[1]:
            score = (param - target[1]) / target[0] * factor
        else:
            score = 0

    if category == 'FI1':
        if param < target[0]:
            score = (target[0] - param) / 25 * factor
        elif param > target[1]:
            score = (param - target[1]) / 25 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'FI2':
        if param < target[0]:
            score = (target[0] - param) / 40 * factor
        elif param > target[1]:
            score = (param - target[1]) / 40 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'FI3':
        if param < target[0]:
            score = (target[0] - param) / 5 * factor
        elif param > target[1]:
            score = (param - target[1]) / 5 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'FI4':
        if param < target[0]:
            score = (target[0] - param) / 40 * factor
        elif param > target[1]:
            score = (param - target[1]) / 40 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'SHW':
        score = (param - target) / target * factor
        if score < 0:
            score = 0
        if score > factor:
            score = factor

    if category == 'AHP':
        if param < target[0]:
            score = (target[0] - param) / 10 * factor
        elif param > target[1]:
            score = (param - target[1]) / 10 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'MHP' or category == 'burst_peak':
        if param < target[0]:
            score = (target[0] - param) / 5 * factor
        elif param > target[1]:
            score = (param - target[1]) / 5 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'IR':
        if param < target[0]:
            score = (target[0] - param) / 100 * factor
        elif param > target[1]:
            score = (param - target[1]) / 100 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'HPC':
        if param < target[0]:
            score = (target[0] - param) / 2 * factor
        elif param > target[1]:
            score = (param - target[1]) / 2 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'PRC':
        if param < target[0]:
            score = (target[0] - param) / 0.02 * factor
        elif param > target[1]:
            score = (param - target[1]) / 0.02 * factor
        else:
            score = 0
        if score > factor:
            score = factor

    if category == 'Rest' or category == 'AP Peak':
        if param <= target[0]:
            score = (target[0] - param) / 5 * factor
        elif param >= target[1]:
            score = (param - target[1]) / 5 * factor
        elif target[0] < param < target[1]:
            score = 0
        else:
            score = factor
        if score > factor:
            score = factor

    if score < 0:
        score = 0
    return score


def cal_AP_width(time, voltage):
    trace1 = {}
    trace1['T'] = time
    trace1['V'] = voltage
    trace1['stim_start'] = [500]
    trace1['stim_end'] = [1000]
    traces = [trace1]
    traces_results = efel.getFeatureValues(traces, ['AP2_width'])
    for trace_results in traces_results:
        # trace_result is a dictionary, with as keys the requested eFeatures
        for feature_name, feature_values in trace_results.items(): 
            if feature_name == 'AP2_width':
                try:
                    spike_half_width = np.mean(feature_values)
                except:
                    spike_half_width = 10
    return spike_half_width


def check_AHP(v):
    peaks = find_peaks(v, height=-20)
    ahp_pc = 0
    if len(peaks[0]) <= 2:
        v_AHP = np.min(v)
        h_pc = np.min(v)
    else:
        d_t = int((peaks[0][1]-peaks[0][0])/4)
        v_check = v[peaks[0][0]:peaks[0][0]+d_t]
        peaks_ahp = find_peaks(v_check)
        ahp_pc = len(peaks_ahp[0])
        if ahp_pc > 0:
            v_AHP = min(v_check[:peaks_ahp[0][0]])
            v_pc = v_check[peaks_ahp[0][0]]
            h_pc = v_pc - v_AHP
        else:
            v_AHP = min(v)
            h_pc = -5
    return v_AHP, ahp_pc, h_pc


def check_peak(v):
    v_peak = max(v)
    return v_peak


def check_rest(v):
    peaks = find_peaks(v, height=0)
    if len(peaks[0]) < 1:
        v_rest = np.mean(v)
    else:
        intervals = peaks[0]
        checkpoints = []
        for i in range(len(intervals)-1):
            point = (intervals[i]+intervals[i+1])/2
            v_r = v[int(point)]
            checkpoints.append(v_r)
        v_rest = np.mean(checkpoints)
    return v_rest


def report_input_impedance(stn_cell, temp):
    h.celsius = temp
    h.finitialize()
    z = h.Impedance()
    z.loc(0.5, sec = stn_cell.soma)
    z.compute(0)
    z_value = z.input(0.5, sec = stn_cell.soma)
    return z_value


def run_cost_simulation(f_index, plotting=False):
    # st = time.time()
    f = tuple(f_index)
    set_aCSF(3)
    dt = 0.025
    h.dt = dt

    # Check input resistance
    f_r = f_index
    f_r[6] = 0
    f_r[7] = 0
    f_r[13] = 0
    f_r[14] = 0
    f_r[21] = 0
    f_r[22] = 0
    stn_cell = create_updated_cell(f_r)

    z_value = report_input_impedance(stn_cell, 37)
    score_ir = 0
    if math.isnan(score_ir):
        score_ir = 100
    print('input resistence = {}'.format(z_value))

    stn_cell = create_updated_cell(f)
    soma_v = h.Vector().record(stn_cell.soma(0.5)._ref_v)
    soma_t = h.Vector().record(h._ref_t)
    h.celsius = 37
    h.finitialize()
    h.continuerun(1500 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_0 = t[int(500/dt):]
    v_0 = v[int(500/dt):]
    soma_v.clear()
    soma_t.clear()
    freq_sp_37 = get_freq_detect_burst(v_0, dt)
    score_sp = cal_score(freq_sp_37, [5, 20], 50, 'SP')
    v_min = min(v_0)

    p_prc = max(np.diff(v_0)/dt)
    score_prc = cal_score(p_prc, [200, 400], 25, 'SP')
    score_sp+=score_prc

    if math.isnan(score_sp):
        score_sp = 100

    # Save State
    svstate = h.SaveState()
    svstate.save()

    # Check AP shape
    # if freq_sp != 0:
    shw = cal_AP_width(t_0, v_0)
    if math.isnan(shw):
        score_shw = 50
    else: 
        score_shw = cal_score(shw, 1, 50, 'SHW')

    v_AHP, ahp_pc, h_pc = check_AHP(v_0)
    score_AHP = cal_score(v_AHP, [-65, -60], 50, 'AHP')
    score_hpc = cal_score(h_pc, [0, 1], 50, 'HPC')
    if ahp_pc < 1:
        score_AHP += 50
    else:
        score_AHP += score_hpc
    if math.isnan(score_AHP):
        score_AHP = 100

    v_rest = check_rest(v_0)
    if math.isnan(v_rest):
        score_rest = 50
    else:
        score_rest = cal_score(v_rest, [-65, -55], 50, 'Rest')

    score_MHP, v_MHP = cal_score_MHP(v_0, 25, [-70, -60])
    v_AM = v_AHP - v_MHP
    score_AM = cal_score(v_AM, [0, 2], 25, 'HPC')
    score_MHP += score_AM
    if math.isnan(score_MHP):
        score_MHP = 50

    v_peak = check_peak(v_0)
    if math.isnan(v_peak):
        score_peak = 50
    else:
        score_peak = cal_score(v_peak, [0, 40], 50, 'AP Peak')

    # Check hyperpolarization current injection
    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1600
    stim.dur = 500
    stim.amp = -0.1
    h.continuerun(2500 * ms)
    v_3 = soma_v.to_python()
    t_3 = soma_t.to_python()
    soma_v.clear()
    soma_t.clear()
    score_hp, v_HP, e_slp, v_half_hp, burst_hp, score_bp, burst_time, burst_count, max_int, min_int = cal_score_HP(v_3, 200, freq_sp_37, dt, v_peak, v_min)
    if math.isnan(score_hp):
        score_hp = 100

    # Check FI curve
    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.1
    h.continuerun(2000 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_1 = t[int(100/dt):]
    v_1 = v[int(100/dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi1 = get_freq_detect_burst(v_1, dt)
    score_fi1 = cal_score(freq_fi1, [65, 75], 75, 'FI1')
    # score_fi1 = cal_score(freq_fi1, [50, 90], 75, 'FI1')
    if math.isnan(score_fi1):
        score_fi1 = 50

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.16
    h.continuerun(2000 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_2 = t[int(100/dt):]
    v_2 = v[int(100/dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi2 = get_freq_detect_burst(v_2, dt)
    score_fi2 = cal_score(freq_fi2, [116, 140], 150, 'FI2')
    # score_fi2 = cal_score(freq_fi2, [105, 140], 150, 'FI2')
    if math.isnan(score_fi2):
        score_fi2 = 50

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.04
    h.continuerun(2000 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_4 = t[int(100/dt):]
    v_4 = v[int(100/dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi3 = get_freq_detect_burst(v_4, dt)
    score_fi3 = cal_score(freq_fi3, [24, 40], 50, 'FI3')
    # score_fi3 = cal_score(freq_fi3, [15, 55], 50, 'FI3')
    if math.isnan(score_fi3):
        score_fi3 = 50

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.2
    h.continuerun(2000 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_4 = t[int(100 / dt):]
    v_4 = v[int(100 / dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi4 = get_freq_detect_burst(v_4, dt)
    score_fi4 = cal_score(freq_fi4, [140, 180], 50, 'FI4')

    svstate.restore()
    stim = h.IClamp(stn_cell.soma(0.5))
    stim.delay = 1500
    stim.dur = 1000
    stim.amp = 0.25
    h.continuerun(2000 * ms)
    v = soma_v.to_python()
    t = soma_t.to_python()
    t_5 = t[int(100 / dt):]
    v_5 = v[int(100 / dt):]
    soma_v.clear()
    soma_t.clear()
    freq_fi5 = get_freq_detect_burst(v_5, dt)
    score_fi4 = cal_score(freq_fi5, [125, 200], 50, 'FI4')

    score_fi = score_fi1 + score_fi2 + score_fi3 + score_fi4
    freq_fi = [freq_fi1, freq_fi2, freq_fi3, freq_fi5]

    # Check extreme HP condition
    check_spikes_after = 0
    max_interval = burst_time
    min_interval = burst_time
    if score_hp < 100 and burst_count > 1:
        svstate.restore()
        stim = h.IClamp(stn_cell.soma(0.5))
        stim.delay = 1600
        stim.dur = 500
        stim.amp = -0.25
        h.continuerun(2500 * ms)
        v_check = soma_v.to_python()
        t_check = soma_t.to_python()
        soma_v.clear()
        soma_t.clear()
        score_hp, check_spikes_after = check_hp_last_spikes(v_check, score_hp, 50, dt)
        score_hp, max_interval, min_interval, bt02 = check_dp_block(v_check, score_hp, 50, freq_sp_37, dt, threshold=-20)

    # find last two peaks
    stn_cell = create_updated_cell(f)
    soma_v = h.Vector().record(stn_cell.soma(0.5)._ref_v)
    soma_t = h.Vector().record(h._ref_t)
    h.finitialize()
    h.continuerun(1500 * ms)
    v_check = soma_v.to_python()
    t_check = soma_t.to_python()
    soma_v.clear()
    soma_t.clear()
    peaks = find_peaks(v_check[int(1000 / dt):int(1500 / dt)], height=0)
    peak_times = (peaks[0] + int(1000 / dt)) * dt
    last_two_peaks_time = peak_times[-2:]

    if len(peak_times) > 2 and len(peak_times) < 12:
        # Add IClamps for PRC
        v_shifts = []
        iprcs = []
        i_start = last_two_peaks_time[-2]
        i_end = last_two_peaks_time[-1]
        i_max = 0.1
        i_step = 0.05 * (i_end - i_start)  # 5% interval
        x_ticks = np.arange(0.05, 0.8, 0.05)
        i_values = [i_start + i * i_step for i in range(int((i_end - i_start) / i_step) + 1)]
        for i_time in i_values[1:16]:
            stn_cell = create_updated_cell(f)
            stim = h.IClamp(stn_cell.dend1(0.5))
            stim.delay = 0
            stim.dur = 1e9  # Infinite duration to allow time-varying current
            # Define the time-varying current
            t_max = 1500  # ms
            start_time = i_time
            time_points = np.arange(0, t_max + dt, dt)
            peak_current = i_max  # Peak current in nA
            current_values = np.zeros_like(time_points)  # Initialize to 0
            current_values[time_points >= start_time] = peak_current * np.exp(
                -(time_points[time_points >= start_time] - start_time) / 2.5
            )
            # Create vectors for time and current
            tvec = h.Vector(time_points.tolist())
            ivec = h.Vector(current_values.tolist())
            ivec.play(stim._ref_amp, tvec)
            soma_v = h.Vector().record(stn_cell.soma(0.5)._ref_v)
            soma_t = h.Vector().record(h._ref_t)
            h.finitialize()
            h.continuerun(1500 * ms)
            v_check = soma_v.to_python()
            t_check = soma_t.to_python()
            soma_v.clear()
            soma_t.clear()
            # plt.figure()
            # plt.plot(time_points, current_values, label='Injected Current')
            # # plt.title('Time-varying Current Input')
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Current (nA)')
            # plt.xlim([start_time-8, start_time+16])
            # # plt.savefig('sim_results/Simulated_Current_Pulse.svg')
            # plt.figure()
            # plt.plot(t_check, v_check)
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Potential (mV)')
            # plt.xlim([start_time-8, start_time+16])
            # plt.ylim([-65, -55])
            # # plt.savefig('sim_results/Simulated_vshift.svg')
            # plt.figure()
            # plt.plot(t_check, v_check)
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Potential (mV)')
            # plt.xlim([1000, 1500])
            # # plt.savefig('sim_results/Phase_shift_ISI.svg')
            # # plt.show()

            # Find peaks in the voltage trace
            peaks = find_peaks(v_check[int(1000 / dt):int(1480 / dt)], height=-20)
            peak_indices = peaks[0] + int(1480 / dt)  # offset peak indices to match full v_check indexing
            # Compute inter-spike intervals (in number of samples)
            diff_p = np.diff(peak_indices)
            # EPSP voltage at the perturbed time
            try:
                v_epsp = v_check[int((start_time + 6) / dt)]
                # Compute average "normal" voltage from previous ISIs at the same phase
                v_normal_list = []
                for isi in diff_p[:-1]:
                    time_idx = int((start_time - isi * dt + 6) / dt)
                    v_normal_list.append(v_check[time_idx])
                v_normal_avg = np.mean(v_normal_list)
                # Voltage shift caused by perturbation
                v_shift = v_epsp - v_normal_avg
                # Compute iPRC approximation
                baseline_isi = np.mean(diff_p[:-1])
                iprc = (baseline_isi - diff_p[-1]) / baseline_isi / v_shift
                v_shifts.append(v_shift)
                iprcs.append(iprc)
            except IndexError:
                print('v_check',len(v_check))
                print('index', int((start_time + 8) / dt))
                iprc = 0
                iprcs.append(0)
    else:
        score_prc = 30
        iprcs = [0]*7

    score_total = (score_sp + score_shw  + score_AHP + score_peak + score_fi + score_hp + score_ir +
                   score_rest + score_MHP)

    if plotting:
        plt.figure()
        plt.plot(t_0, v_0, 'k')
        plt.title('Spontaneous Spiking {}'.format(freq_sp_37))
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.ylim([-85, 25])
        # plt.savefig('sim_results/SP.svg')

        plt.figure()
        peaks = find_peaks(v_0, height=0)
        t_peak = peaks[0][1]
        v_single = v_0[t_peak - int(10 / h.dt):t_peak + int(15 / h.dt)]
        t_single = t_0[t_peak - int(10 / h.dt):t_peak + int(15 / h.dt)]
        plt.plot(t_single, v_single)
        plt.title('Single AP')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.ylim([-85, 25])
        # plt.savefig('sim_results/single_ap.svg')

        plt.figure()
        plt.plot(v_0[1:], np.diff(v_0)/dt, 'k')
        # plt.savefig('sim_results/SP_diff.svg')


        plt.figure()
        plt.plot(t_3, v_3, 'k')
        plt.xlabel('Time (ms)')
        plt.ylabel('Potential (mV)')
        plt.ylim([-85, 25])
        # plt.savefig('sim_results/HP.svg')

        plt.figure()
        I = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25]
        Upper = [17.9644517, 29.1063352, 40.40744386, 54.99848476, 73.24940317, 99.18888674, 127.2044039, 144.5001211, 163.5276886, 175, 190, 212]
        Lower = [8.4355483, 14.8936648, 21.59255614, 28.60151524, 35.55059683, 41.61111326, 49.99559611, 61.89987893, 80.07231135, 93, 121, 153]
        plt.fill_between(I, Upper, Lower, color='0.8')
        plt.plot([0, 0.04, 0.1, 0.16, 0.2, 0.25], [freq_sp_37, freq_fi3, freq_fi1, freq_fi2, freq_fi4, freq_fi5], 'k')
        plt.title('FI Curve')
        plt.xlabel('Current Injected (nA)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim([-5, 235])
        plt.xticks(np.arange(min(I), max(I)+0.05, 0.05))
        plt.xlim([-0.01, 0.26])
        # plt.savefig('sim_results/FI Curve.svg')

        plt.figure()
        plt.plot(x_ticks, iprcs, 'k')
        plt.ylim([-0.01, 0.1])
        # plt.savefig('sim_results/iPRC_Curve_full.svg')

        plt.show()

    return [score_total, freq_sp_37, p_prc, score_shw, v_AHP, score_AHP, v_MHP, v_AM, freq_fi3, freq_fi1, freq_fi2, freq_fi5,
            v_HP, e_slp, v_half_hp, burst_hp, score_bp, burst_time, burst_count, check_spikes_after, z_value, v_rest,
            max_interval, min_interval, max_int, min_int]


if __name__ == "__main__":
    with open('MatingPool_final_sorted.pickle','rb') as p_file:
        MatingPool = pickle.load(p_file)

    pool_index = 9

    f = MatingPool[0][pool_index]
    index_score = MatingPool[1][pool_index]
    print(f'Score Index Picked = {index_score:.2f}')
    
    # index_min = MatingPool[1].index(min(MatingPool[1]))
    # print('Score Min = {}'.format(MatingPool[1][index_min]))
    # print('Index Min = {}'.format(index_min))

    # print(f)

    scores = run_cost_simulation(f, plotting=True)
    print(f'simulation score: {scores[0]:.2f}')
