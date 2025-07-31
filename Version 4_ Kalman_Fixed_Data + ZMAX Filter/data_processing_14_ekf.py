import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools



FILE_PATH = './Synched_Data_GR0_22_DEN_MAXZ1_25/NEWDATA/'
FILE_DATES = ['101922', '102122', '111422', '111622', '120522', '120722', '013023', '020123', '031323', '031523', '041723', '041923', '061523']
TARGET_SUBJECT = "DS_STARFISH_2223_27"
DT_VIRT = 0.5
SIGMA_MIN = 1e-6
SIGMA_MAX = 10
D_INITIAL = 0.23
param_grid = list(itertools.product(
    [0.01, 0.05, 0.1, 0.5],  # sigma_v
    [0.001, 0.01, 0.05, 0.1],  # sigma_omega
    [0.5, 1.0]  # sigma_obs
))



def load_data(date_index=0):
    file_name = f'DAYUBIGR_{FILE_DATES[date_index]}_GR0_22_DEN_032825_V2392628911.CSV'
    full_path = FILE_PATH + file_name

    df = pd.read_csv(full_path, header=None, names=['SUBJECTID', 'TIME', 'X', 'Y', 'Z'])
    df = df[(df["X"] <= 15) & (df["Y"] <= 9) & (df["X"] >= 0) & (df["Y"] >= 0)].copy()
    df = df[df['SUBJECTID'].str.startswith(TARGET_SUBJECT)].copy()
    
    df['TIME'] = pd.to_datetime(df['TIME'])
    df['timestamp'] = (df['TIME'] - df['TIME'].min()).dt.total_seconds()
    df['side'] = df['SUBJECTID'].str.extract(r'(\d+[LR])$')[0].str[-1].map({'L': 'left', 'R': 'right'})
    df['timestamp_rounded'] = df['timestamp'].round(3)
    
    return df



def build_observation_data(df):
    grouped = df.groupby('timestamp_rounded')
    data = []

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
        
        data.append(entry)
    
    return data



def add_noise(params, std=0.075):
    noisy = [p * (1 + np.random.normal(0, std)) for p in params]
    noisy[0] = np.clip(noisy[0], SIGMA_MIN, SIGMA_MAX)  # sigma_v
    noisy[1] = np.clip(noisy[1], SIGMA_MIN, SIGMA_MAX)  # sigma_omega
    noisy[2] = np.clip(noisy[2], 0.01, SIGMA_MAX)       # sigma_obs
    noisy[3] = np.clip(noisy[3], 0.01, 1.0)              # d
    return noisy



def track_parameters(params): 
    global param_history, iteration_count
    
    sigma_v, sigma_omega, sigma_obs, d = params
    
    param_history['iteration'].append(iteration_count)
    param_history['sigma_v'].append(sigma_v)
    param_history['sigma_omega'].append(sigma_omega)
    param_history['sigma_obs'].append(sigma_obs)
    param_history['d'].append(d)
    
    iteration_count += 1
    
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
        # Lx, Ly, Rx, Ry
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
    else:  # 'none'
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
    else:  # 'none'
        return np.zeros((0, 6))
    
def ekf_forward(data, timestamps, virtual_timestamps, params):
    sigma_v, sigma_omega, sigma_obs, d = params
    master_timestamps = sorted(set(timestamps + virtual_timestamps))
    T = len(master_timestamps)
    s_hat = [np.zeros(6)] * T  # Predicted states
    P = [np.zeros((6, 6))] * T  # Predicted covariances
    s_filt = [np.zeros(6)] * T  # Filtered states
    P_filt = [np.zeros((6, 6))] * T  # Filtered covariances
    neg_log_likelihood = 0.0  # Initialize negative log-likelihood for minimization

    # Initialize state and covariance
    s_hat[0] = np.zeros(6)
    # Try to initialize with first observation if available
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
                
    P[0] = np.diag([10, 10, 10, 5, 5, 2])  # Initial uncertainty
    s_filt[0] = s_hat[0]  # Initialize filtered state with predicted state
    P_filt[0] = P[0]      # Initialize filtered covariance with predicted covariance

    for k in range(T - 1):
        t_k = master_timestamps[k]
        t_k1 = master_timestamps[k + 1]
        delta_t = t_k1 - t_k

        # Prediction step
        s_hat[k + 1] = state_transition(s_filt[k], delta_t)
        F_k = jacobian_F(delta_t)
        # Modified process noise: same sigma_v for both vx and vy
        Q_k = np.diag([0, 0, 0, sigma_v**2 * delta_t, sigma_v**2 * delta_t, sigma_omega**2 * delta_t])
        P[k + 1] = F_k @ P_filt[k] @ F_k.T + Q_k

        # Update step (only for actual timestamps with observations)
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
                
                # Ensure S_k1 is positive definite for numerical stability
                S_k1 = (S_k1 + S_k1.T) / 2  # Make symmetric
                
                try:
                    # Calculate innovation and its contribution to negative log likelihood
                    innovation = z_k1 - z_pred
                    sign, logdet = np.linalg.slogdet(S_k1)
                    if sign > 0:  # Check if determinant is positive
                        # Use log determinant formula for numerical stability
                        neg_log_likelihood += 0.5 * (m_t * np.log(2 * np.pi) + logdet + 
                                              innovation @ np.linalg.inv(S_k1) @ innovation)
                    else:
                        print(sign, logdet)
                        print('S_k1', S_k1)
                        raise np.linalg.LinAlgError("S_k1 is not positive definite")
                    # Kalman gain and state update
                    K_k1 = P[k + 1] @ H_k1.T @ np.linalg.inv(S_k1)
                    s_filt[k + 1] = s_hat[k + 1] + K_k1 @ innovation
                    P_filt[k + 1] = (np.eye(6) - K_k1 @ H_k1) @ P[k + 1]
                except np.linalg.LinAlgError:
                    print('parameters:', params)
                    print('S_k1', S_k1)
                    print('Innovation:', innovation)
                    print('P[k + 1]:', P[k + 1])
                    print('H_k1:', H_k1)
                    print('z_pred:', z_pred)
                    print('z_k1:', z_k1)
                    print('s_hat[k + 1]:', s_hat[k + 1])
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
        t_k = master_timestamps[k]
        t_k1 = master_timestamps[k + 1]
        delta_t = t_k1 - t_k
        F_k = jacobian_F(delta_t)
        
        try:
            C_k = P_filt[k] @ F_k.T @ np.linalg.inv(P[k + 1])
            s_smooth[k] = s_filt[k] + C_k @ (s_smooth[k + 1] - s_hat[k + 1])
            P_smooth[k] = P_filt[k] + C_k @ (P_smooth[k + 1] - P[k + 1]) @ C_k.T
        except np.linalg.LinAlgError:
            s_smooth[k] = s_filt[k]
            P_smooth[k] = P_filt[k]

    return s_smooth, P_smooth



def optimize_params(data, timestamps, virtual_ts, init_params):
    def objective(params):
        try:
            _, _, _, _, nll = ekf_forward(data, timestamps, virtual_ts, params)
            return nll
        except:
            return np.inf

    bounds = [(SIGMA_MIN, SIGMA_MAX), (SIGMA_MIN, SIGMA_MAX), (0.01, SIGMA_MAX), (0.01, 1.0)]
    result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20})
    return result.x



def run_optimization_loop(real_data):
    np.random.seed(42)
    results = []

    timestamps = [e['timestamp'] for e in real_data[:3000]]
    virtual_ts = list(np.arange(min(timestamps), max(timestamps), DT_VIRT))

    for i, (sigma_v, sigma_omega, sigma_obs) in enumerate(param_grid):
        try:
            init_params = [sigma_v, sigma_omega, sigma_obs, D_INITIAL]
            best_params = init_params
            best_nll = np.inf
            current = init_params

            for _ in range(10):
                opt_params = optimize_params(real_data[:3000], timestamps, virtual_ts, current)
                _, _, _, _, nll = ekf_forward(real_data[:3000], timestamps, virtual_ts, opt_params)

                if nll < best_nll:
                    best_params = opt_params
                    best_nll = nll

                current = add_noise(opt_params)

            results.append({
                'combination_id': i + 1,
                'initial': init_params,
                'final': best_params,
                'nll': best_nll,
                'success': True
            })

        except Exception as e:
            results.append({
                'combination_id': i + 1,
                'initial': [sigma_v, sigma_omega, sigma_obs, D_INITIAL],
                'final': [np.nan]*4,
                'nll': np.inf,
                'success': False
            })

    return results



def segment_time(data, threshold=15.0):
    df = pd.DataFrame(data).sort_values('timestamp')
    df['diff'] = df['timestamp'].diff()
    segments = []
    start = df['timestamp'].iloc[0]

    for _, row in df[df['diff'] > threshold].iterrows():
        end = row['timestamp'] - row['diff']
        seg_data = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        segments.append({'start': start, 'end': end, 'duration': end - start, 'num_points': len(seg_data)})
        start = row['timestamp']

    final_data = df[df['timestamp'] >= start]
    segments.append({'start': start, 'end': df['timestamp'].iloc[-1], 'duration': df['timestamp'].iloc[-1] - start, 'num_points': len(final_data)})
    return pd.DataFrame(segments)



def main():
    df = load_data(date_index=0)
    real_data = build_observation_data(df)
    results = run_optimization_loop(real_data)
    
    results_df = pd.DataFrame(results)
    output_file = 'ekf_optimization_results_cleaned.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    
    # Get best result
    success_results = [r for r in results if r['success']]
    if success_results:
        best_result = min(success_results, key=lambda r: r['nll'])
        print(f"\nBest params: {best_result['final']} with NLL: {best_result['nll']:.4f}")
    
    return results_df

if __name__ == "__main__":
    main()
