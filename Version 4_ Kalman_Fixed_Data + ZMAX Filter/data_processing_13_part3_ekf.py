import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# ---------------------- File Setup ----------------------

file_path = './Synched_Data_GR0_22_DEN_MAXZ1_25/NEWDATA/'
file_date = ['101922', '102122', '111422', '111622', '120522', '120722', '013023', '020123', '031323', '031523', '041723', '041923', '061523']
date = file_date[0]

file_name = f'DAYUBIGR_{date}_GR0_22_DEN_032825_V2392628911.CSV'
full_path = file_path + file_name

# ---------------------- Data Loading and Preprocessing ----------------------

raw_data = pd.read_csv(full_path, header=None, names=['SUBJECTID', 'TIME', 'X', 'Y', 'Z'])
clear_data = raw_data.reset_index(drop=True)
clear_data = clear_data[(clear_data["X"] <= 15) & (clear_data["Y"] <= 9) & 
                        (clear_data["X"] >= 0) & (clear_data["Y"] >= 0)].copy()

target_subject_base = "DS_STARFISH_2223_27"
subject_data = clear_data[clear_data['SUBJECTID'].str.startswith(target_subject_base)].copy()
subject_data['TIME'] = pd.to_datetime(subject_data['TIME'])
t0 = subject_data['TIME'].min()
subject_data['timestamp'] = (subject_data['TIME'] - t0).dt.total_seconds()

subject_data['side'] = subject_data['SUBJECTID'].str.extract(r'(\d+[LR])$')[0].str[-1].map({'L': 'left', 'R': 'right'})
subject_data['timestamp_rounded'] = subject_data['timestamp'].round(3)

# Organize data into time-stamped entries
grouped = subject_data.groupby('timestamp_rounded')
real_data = []

for ts, group in grouped:
    entry = {'timestamp': ts}
    left = group[group['side'] == 'left']
    right = group[group['side'] == 'right']
    
    if not left.empty:
        entry['left'] = left[['X', 'Y']].iloc[0].to_numpy()
    if not right.empty:
        entry['right'] = right[['X', 'Y']].iloc[0].to_numpy()
    
    if 'left' in entry and 'right' in entry:
        entry['observed'] = 'both'
        entry['obs'] = np.concatenate([entry['left'], entry['right']])
    elif 'left' in entry:
        entry['observed'] = 'left'
        entry['obs'] = entry['left']
    elif 'right' in entry:
        entry['observed'] = 'right'
        entry['obs'] = entry['right']
    else:
        entry['observed'] = 'none'
        entry['obs'] = np.array([])

    real_data.append(entry)

# ---------------------- EKF and Smoother Functions ----------------------

def state_transition(s_t, delta_t):
    x, y, theta, vx, vy, omega = s_t
    return np.array([
        x + vx * delta_t,
        y + vy * delta_t,
        theta + omega * delta_t,
        vx,
        vy,
        omega
    ])

def jacobian_F(delta_t):
    return np.array([
        [1, 0, 0, delta_t, 0, 0],
        [0, 1, 0, 0, delta_t, 0],
        [0, 0, 1, 0, 0, delta_t],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

def h(s_t, observed_sensors, d):
    x, y, theta = s_t[0], s_t[1], s_t[2]
    if observed_sensors == 'both':
        return np.array([
            x - d * np.sin(theta),
            y + d * np.cos(theta),
            x + d * np.sin(theta),
            y - d * np.cos(theta)
        ])
    elif observed_sensors == 'left':
        return np.array([x - d * np.sin(theta), y + d * np.cos(theta)])
    elif observed_sensors == 'right':
        return np.array([x + d * np.sin(theta), y - d * np.cos(theta)])
    else:
        return np.array([])

def jacobian_h(s_t, observed_sensors, d):
    theta = s_t[2]
    if observed_sensors == 'both':
        return np.array([
            [1, 0, -d * np.cos(theta), 0, 0, 0],
            [0, 1, -d * np.sin(theta), 0, 0, 0],
            [1, 0, d * np.cos(theta), 0, 0, 0],
            [0, 1, d * np.sin(theta), 0, 0, 0]
        ])
    elif observed_sensors == 'left':
        return np.array([
            [1, 0, -d * np.cos(theta), 0, 0, 0],
            [0, 1, -d * np.sin(theta), 0, 0, 0]
        ])
    elif observed_sensors == 'right':
        return np.array([
            [1, 0, d * np.cos(theta), 0, 0, 0],
            [0, 1, d * np.sin(theta), 0, 0, 0]
        ])
    else:
        return np.zeros((0, 6))

def ekf_forward(data, timestamps, virtual_timestamps, params):
    sigma_vx, sigma_vy, sigma_omega, sigma_obs, d = params
    master_timestamps = sorted(set(timestamps + virtual_timestamps))
    T = len(master_timestamps)
    s_hat = [np.zeros(6)] * T
    P = [np.zeros((6, 6))] * T
    s_filt = [np.zeros(6)] * T
    P_filt = [np.zeros((6, 6))] * T
    neg_log_likelihood = 0.0

    for entry in data[:10]:
        if entry['observed'] != 'none':
            if entry['observed'] == 'left':
                s_hat[0][:2] = entry['left']
                break
            elif entry['observed'] == 'right':
                s_hat[0][:2] = entry['right']
                break
            elif entry['observed'] == 'both':
                s_hat[0][:2] = (entry['left'] + entry['right']) / 2
                break

    P[0] = np.diag([10, 10, 10, 5, 5, 2])
    s_filt[0] = s_hat[0]
    P_filt[0] = P[0]

    for k in range(T - 1):
        t_k, t_k1 = master_timestamps[k], master_timestamps[k + 1]
        delta_t = t_k1 - t_k

        s_hat[k + 1] = state_transition(s_filt[k], delta_t)
        F_k = jacobian_F(delta_t)
        Q_k = block_diag(0, 0, 0, sigma_vx**2 * delta_t**2, sigma_vy**2 * delta_t**2, sigma_omega**2 * delta_t**2)
        P[k + 1] = F_k @ P_filt[k] @ F_k.T + Q_k

        if t_k1 in timestamps:
            idx = timestamps.index(t_k1)
            observed_sensors = data[idx]['observed']
            if observed_sensors != 'none':
                H_k1 = jacobian_h(s_hat[k + 1], observed_sensors, d)
                z_pred = h(s_hat[k + 1], observed_sensors, d)
                z_k1 = data[idx]['obs']
                m_t = len(z_k1)
                R = sigma_obs**2 * np.eye(m_t)
                S_k1 = H_k1 @ P[k + 1] @ H_k1.T + R
                S_k1 = (S_k1 + S_k1.T) / 2

                try:
                    innovation = z_k1 - z_pred
                    sign, logdet = np.linalg.slogdet(S_k1)
                    if sign > 0:
                        neg_log_likelihood += 0.5 * (m_t * np.log(2 * np.pi) + logdet +
                                                     innovation @ np.linalg.inv(S_k1) @ innovation)
                    else:
                        raise np.linalg.LinAlgError("S_k1 is not positive definite")

                    K_k1 = P[k + 1] @ H_k1.T @ np.linalg.inv(S_k1)
                    s_filt[k + 1] = s_hat[k + 1] + K_k1 @ innovation
                    P_filt[k + 1] = (np.eye(6) - K_k1 @ H_k1) @ P[k + 1]
                except np.linalg.LinAlgError:
                    raise
            else:
                s_filt[k + 1] = s_hat[k + 1]
                P_filt[k + 1] = P[k + 1]
        else:
            s_filt[k + 1] = s_hat[k + 1]
            P_filt[k + 1] = P[k + 1]

    return s_filt, P_filt, s_hat, P, neg_log_likelihood

def smoother(s_filt, P_filt, s_hat, P, timestamps, virtual_timestamps):
    master_timestamps = sorted(set(timestamps + virtual_timestamps))
    T = len(master_timestamps)
    s_smooth = [np.zeros(6)] * T
    P_smooth = [np.zeros((6, 6))] * T
    s_smooth[-1] = s_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for k in range(T - 2, -1, -1):
        delta_t = master_timestamps[k + 1] - master_timestamps[k]
        F_k = jacobian_F(delta_t)
        try:
            C_k = P_filt[k] @ F_k.T @ np.linalg.inv(P[k + 1])
            s_smooth[k] = s_filt[k] + C_k @ (s_smooth[k + 1] - s_hat[k + 1])
            P_smooth[k] = P_filt[k] + C_k @ (P_smooth[k + 1] - P[k + 1]) @ C_k.T
        except np.linalg.LinAlgError:
            s_smooth[k] = s_filt[k]
            P_smooth[k] = P_filt[k]

    return s_smooth, P_smooth

# ---------------------- Time Segments ----------------------

DT_VIRT = 0.5
df = pd.DataFrame(real_data).sort_values('timestamp')
df['time_diff'] = df['timestamp'].diff()

gaps = df[df['time_diff'] > 15.0]
segments = []
current_start = df['timestamp'].iloc[0]

for _, row in gaps.iterrows():
    end = row['timestamp'] - row['time_diff']
    segment_data = df[(df['timestamp'] >= current_start) & (df['timestamp'] <= end)]
    segments.append({'start': current_start, 'end': end, 'duration': end - current_start, 'num_points': len(segment_data)})
    current_start = row['timestamp']

final_segment = df[(df['timestamp'] >= current_start)]
segments.append({'start': current_start, 'end': df['timestamp'].iloc[-1], 'duration': df['timestamp'].iloc[-1] - current_start, 'num_points': len(final_segment)})

time_segments = []
for segment in segments:
    seg_data = df[(df['timestamp'] >= segment['start']) & (df['timestamp'] <= segment['end'])].copy()
    time_segments.append({**segment, 'data': seg_data})

# ---------------------- Final EKF Run and Plot ----------------------

# final_params = [0.06787, 0.08011, 0.04829, 0.38712, 0.41107]

# plt.figure(figsize=(14, 10))
# plt.title('Final EKF Results with Best Parameters')

# left_x, left_y, right_x, right_y = [], [], [], []
# for entry in real_data:
#     if 'left' in entry:
#         left_x.append(entry['left'][0])
#         left_y.append(entry['left'][1])
#     if 'right' in entry:
#         right_x.append(entry['right'][0])
#         right_y.append(entry['right'][1])
# plt.scatter(left_x, left_y, color='blue', alpha=0.3, label='Left Sensor', s=10)
# plt.scatter(right_x, right_y, color='red', alpha=0.3, label='Right Sensor', s=10)

# first = True
# for seg in time_segments:
#     run_data = seg['data'].to_dict(orient='records')
#     timestamps = [entry['timestamp'] for entry in run_data]
#     virtual_ts = np.arange(min(timestamps), max(timestamps), DT_VIRT).tolist()

#     s_filt, P_filt, s_hat, P, _ = ekf_forward(run_data, timestamps, virtual_ts, final_params)
#     s_smooth, _ = smoother(s_filt, P_filt, s_hat, P, timestamps, virtual_ts)

#     label = 'Smoothed Trajectory' if first else None
#     first = False

#     plt.plot([s[0] for s in s_smooth], [s[1] for s in s_smooth], 'g-', alpha=0.8, linewidth=2, label=label)

# plt.grid(True, alpha=0.3)
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.legend()
# plt.show()