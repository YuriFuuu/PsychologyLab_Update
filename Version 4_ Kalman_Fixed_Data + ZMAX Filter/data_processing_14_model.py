import numpy as np
import pandas as pd
from scipy.stats import chi2
from data_processing_14_ekf import ekf_forward, smoother


def prepare_real_data(subject_data, dt_virt=0.5):
    subject_data['TIME'] = pd.to_datetime(subject_data['TIME'])
    subject_data['timestamp'] = (subject_data['TIME'] - subject_data['TIME'].min()).dt.total_seconds()
    subject_data['side'] = subject_data['SUBJECTID'].str.extract(r'(\d+[LR])$')[0].str[-1].map({'L': 'left', 'R': 'right'})
    subject_data['timestamp_rounded'] = subject_data['timestamp'].round(3)

    grouped = subject_data.groupby('timestamp_rounded')
    real_data = []
    for ts, group in grouped:
        entry = {'timestamp': ts}
        for side in ['left', 'right']:
            side_data = group[group['side'] == side]
            if not side_data.empty:
                entry[side] = side_data[['X', 'Y']].iloc[0].to_numpy()
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

    df = pd.DataFrame(real_data).sort_values('timestamp')
    df['time_diff'] = df['timestamp'].diff()
    segments = []
    current_start = df['timestamp'].iloc[0]

    for _, row in df[df['time_diff'] > 15.0].iterrows():
        end = row['timestamp'] - row['time_diff']
        segment_data = df[(df['timestamp'] >= current_start) & (df['timestamp'] <= end)]
        segments.append({'start': current_start, 'end': end, 'data': segment_data})
        current_start = row['timestamp']

    final_segment = df[df['timestamp'] >= current_start]
    segments.append({'start': current_start, 'end': df['timestamp'].iloc[-1], 'data': final_segment})

    return real_data, segments


def run_ekf_on_segments(segments, final_params, dt_virt=0.5):
    trajectories = []
    for seg in segments:
        run_data = seg['data'].to_dict(orient='records')
        if len(run_data) < 2:
            continue
        timestamps = [e['timestamp'] for e in run_data]
        virtual_ts = np.arange(min(timestamps), max(timestamps), dt_virt).tolist()

        try:
            s_filt, P_filt, s_hat, P, _ = ekf_forward(run_data, timestamps, virtual_ts, final_params)
            s_smooth, P_smooth = smoother(s_filt, P_filt, s_hat, P, timestamps, virtual_ts)
            trajectories.append({
                'start_time': min(timestamps),
                'end_time': max(timestamps),
                'trajectory': s_smooth,
                'covariances': P_smooth,
                'real_ts': timestamps,
                'virtual_ts': virtual_ts,
                'all_ts': np.sort(virtual_ts + timestamps),
                'run_data': run_data
            })
        except Exception as e:
            print(f"Error processing segment: {e}")
    return trajectories


def get_observations(data, up_to_time=None):
    left_x, left_y, right_x, right_y = [], [], [], []
    for entry in data:
        if up_to_time is not None and entry['timestamp'] > up_to_time:
            break
        if 'left' in entry:
            left_x.append(entry['left'][0])
            left_y.append(entry['left'][1])
        if 'right' in entry:
            right_x.append(entry['right'][0])
            right_y.append(entry['right'][1])
    return (left_x, left_y), (right_x, right_y)


def interpolate_trajectory(trajectories, current_time):
    results = []
    current_position = None
    current_covariance = None
    for segment in trajectories:
        if segment['start_time'] > current_time:
            continue
        limit_time = min(current_time, segment['end_time'])
        indices = [i for i, t in enumerate(segment['real_ts']) if t <= limit_time]
        if indices:
            x, y, covs = [], [], []
            for idx in indices:
                t = segment['real_ts'][idx]
                closest_idx = np.argmin(np.abs(np.array(segment['all_ts']) - t))
                if closest_idx < len(segment['trajectory']):
                    x.append(segment['trajectory'][closest_idx][0])
                    y.append(segment['trajectory'][closest_idx][1])
                    if closest_idx < len(segment['covariances']):
                        covs.append(segment['covariances'][closest_idx])
            if x and y:
                results.append({'x': x, 'y': y, 'covariances': covs})
                if segment['end_time'] >= current_time or segment == trajectories[-1]:
                    current_position = (x[-1], y[-1])
                    current_covariance = covs[-1] if covs else None
    return results, current_position, current_covariance


def build_uncertainty_ellipse(position, covariance, level):
    pos_cov = covariance[:2, :2]
    eigvals, eigvecs = np.linalg.eigh(pos_cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    chi2_val = chi2.ppf(level, df=2)
    width, height = 2 * np.sqrt(chi2_val * eigvals)
    return {
        'center': position,
        'width': width,
        'height': height,
        'angle': angle
    }