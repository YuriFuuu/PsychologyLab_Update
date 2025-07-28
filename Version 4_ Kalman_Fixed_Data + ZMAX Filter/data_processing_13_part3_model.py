# import numpy as np
# import pandas as pd
# from scipy.stats import chi2
# from data_processing_13_part3_ekf import ekf_forward, smoother

# def compute_smoothed_trajectory(subject_data, final_params, dt_virt=0.5):
#     subject_data['TIME'] = pd.to_datetime(subject_data['TIME'])
#     t0 = subject_data['TIME'].min()
#     subject_data['timestamp'] = (subject_data['TIME'] - t0).dt.total_seconds()
    
#     subject_data['side'] = subject_data['SUBJECTID'].str.extract(r'(\d+[LR])$')[0].str[-1].map({'L': 'left', 'R': 'right'})
#     subject_data['timestamp_rounded'] = subject_data['timestamp'].round(3)
    
#     grouped = subject_data.groupby('timestamp_rounded')
#     real_data = []
#     for ts, group in grouped:
#         entry = {'timestamp': ts}
#         left = group[group['side'] == 'left']
#         right = group[group['side'] == 'right']
#         if not left.empty:
#             entry['left'] = left[['X', 'Y']].iloc[0].to_numpy()
#         if not right.empty:
#             entry['right'] = right[['X', 'Y']].iloc[0].to_numpy()
#         if 'left' in entry and 'right' in entry:
#             entry['observed'] = 'both'
#             entry['obs'] = np.concatenate([entry['left'], entry['right']])
#         elif 'left' in entry:
#             entry['observed'] = 'left'
#             entry['obs'] = entry['left']
#         elif 'right' in entry:
#             entry['observed'] = 'right'
#             entry['obs'] = entry['right']
#         else:
#             entry['observed'] = 'none'
#             entry['obs'] = np.array([])
#         real_data.append(entry)
    
#     real_data.sort(key=lambda x: x['timestamp'])
    
#     df = pd.DataFrame(real_data).sort_values('timestamp')
#     df['time_diff'] = df['timestamp'].diff()
    
#     gaps = df[df['time_diff'] > 15.0]
#     segments = []
#     current_start = df['timestamp'].iloc[0]
    
#     for _, row in gaps.iterrows():
#         end = row['timestamp'] - row['time_diff']
#         segment_data = df[(df['timestamp'] >= current_start) & (df['timestamp'] <= end)]
#         segments.append({'start': current_start, 'end': end, 'duration': end - current_start, 'num_points': len(segment_data)})
#         current_start = row['timestamp']
    
#     # Add final segment
#     final_segment = df[(df['timestamp'] >= current_start)]
#     segments.append({'start': current_start, 'end': df['timestamp'].iloc[-1], 'duration': df['timestamp'].iloc[-1] - current_start, 'num_points': len(final_segment)})
    
#     time_segments = []
#     for segment in segments:
#         seg_data = df[(df['timestamp'] >= segment['start']) & (df['timestamp'] <= segment['end'])].copy()
#         time_segments.append({'start': segment['start'], 'end': segment['end'], 'data': seg_data})
    
#     return real_data, time_segments


# def process_ekf_segments(time_segments, final_params, dt_virt=0.5):

#     segment_trajectories = []
    
#     for time_segment in time_segments:
#         run_data = time_segment['data'].to_dict(orient='records')
#         if len(run_data) < 2:
#             continue
            
#         timestamps = [entry['timestamp'] for entry in run_data]
#         virtual_ts = np.arange(min(timestamps), max(timestamps), dt_virt).tolist()
        
#         try:
#             s_filt, P_filt, s_hat, P, _ = ekf_forward(run_data, timestamps, virtual_ts, final_params)
#             s_smooth, P_smooth = smoother(s_filt, P_filt, s_hat, P, timestamps, virtual_ts)
            
#             segment_trajectories.append({
#                 'start_time': min(timestamps),
#                 'end_time': max(timestamps),
#                 'virtual_ts': virtual_ts,
#                 'real_ts': timestamps,
#                 'all_ts': np.sort(virtual_ts + timestamps),
#                 'trajectory': s_smooth,
#                 'covariances': P_smooth,
#                 'run_data': run_data
#             })
#         except Exception as e:
#             print(f"Error processing segment: {e}")
#             continue
    
#     return segment_trajectories


# def extract_observations(real_data):

#     left_x, left_y = [], []
#     right_x, right_y = [], []
    
#     for entry in real_data:
#         if 'left' in entry:
#             left_x.append(entry['left'][0])
#             left_y.append(entry['left'][1])
#         if 'right' in entry:
#             right_x.append(entry['right'][0])
#             right_y.append(entry['right'][1])
    
#     return (left_x, left_y), (right_x, right_y)


# def get_current_observations(real_data, current_time):

#     current_left_x, current_left_y = [], []
#     current_right_x, current_right_y = [], []
    
#     for entry in real_data:
#         if entry['timestamp'] <= current_time:
#             if 'left' in entry:
#                 current_left_x.append(entry['left'][0])
#                 current_left_y.append(entry['left'][1])
#             if 'right' in entry:
#                 current_right_x.append(entry['right'][0])
#                 current_right_y.append(entry['right'][1])
#         else:
#             break
    
#     return (current_left_x, current_left_y), (current_right_x, current_right_y)


# def get_trajectory_for_time(segment_trajectories, current_time):

#     trajectory_data = []
#     current_position = None
#     current_covariance = None
    
#     for segment in segment_trajectories:
#         if segment['start_time'] > current_time:
#             continue
            
#         segment_current_time = min(current_time, segment['end_time'])
        
#         # Find real timestamps that are valid for the current segment
#         valid_real_indices = [i for i, t in enumerate(segment['real_ts']) if t <= segment_current_time]
        
#         if valid_real_indices:
#             # Get trajectory points corresponding to real timestamps
#             trajectory_x = []
#             trajectory_y = []
#             trajectory_covariances = []
            
#             for real_idx in valid_real_indices:
#                 # Find the closest virtual timestamp index for interpolation
#                 real_t = segment['real_ts'][real_idx]
                
#                 # Find the closest timestamp
#                 allts_idx = np.argmin(np.abs(np.array(segment['all_ts']) - real_t))
                
#                 if allts_idx is not None and allts_idx < len(segment['trajectory']):
#                     trajectory_x.append(segment['trajectory'][allts_idx][0])
#                     trajectory_y.append(segment['trajectory'][allts_idx][1])
#                     if allts_idx < len(segment['covariances']):
#                         trajectory_covariances.append(segment['covariances'][allts_idx])
            
#             if trajectory_x and trajectory_y:
#                 trajectory_data.append({
#                     'x': trajectory_x,
#                     'y': trajectory_y,
#                     'covariances': trajectory_covariances
#                 })
                
#                 # Store current position uncertainty for later plotting
#                 if segment['end_time'] >= current_time or segment == segment_trajectories[-1]:
#                     current_position = (trajectory_x[-1], trajectory_y[-1])
#                     if trajectory_covariances:
#                         current_covariance = trajectory_covariances[-1]
    
#     return trajectory_data, current_position, current_covariance


# def create_uncertainty_ellipse(position, covariance, confidence_level):

#     pos_cov = covariance[:2, :2]
#     eigenvals, eigenvecs = np.linalg.eigh(pos_cov)
#     order = eigenvals.argsort()[::-1]
#     eigenvals = eigenvals[order]
#     eigenvecs = eigenvecs[:, order]
    
#     # Calculate angle of major axis
#     angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
#     # Chi-square critical value for 2 DOF
#     chi2_val = chi2.ppf(confidence_level, df=2)
    
#     # Calculate ellipse dimensions
#     width = 2 * np.sqrt(chi2_val * eigenvals[0])
#     height = 2 * np.sqrt(chi2_val * eigenvals[1])
    
#     return {
#         'center': position,
#         'width': width,
#         'height': height,
#         'angle': angle
#     }



# Fixed data_processing_13_part3_model.py functions

import numpy as np
import pandas as pd
from scipy.stats import chi2
from data_processing_13_part3_ekf import ekf_forward, smoother

def compute_smoothed_trajectory(subject_data, final_params, dt_virt=0.5):
    subject_data['TIME'] = pd.to_datetime(subject_data['TIME'])
    t0 = subject_data['TIME'].min()
    subject_data['timestamp'] = (subject_data['TIME'] - t0).dt.total_seconds()
    
    subject_data['side'] = subject_data['SUBJECTID'].str.extract(r'(\d+[LR])$')[0].str[-1].map({'L': 'left', 'R': 'right'})
    subject_data['timestamp_rounded'] = subject_data['timestamp'].round(3)
    
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
    
    real_data.sort(key=lambda x: x['timestamp'])
    
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
    
    # Add final segment
    final_segment = df[(df['timestamp'] >= current_start)]
    segments.append({'start': current_start, 'end': df['timestamp'].iloc[-1], 'duration': df['timestamp'].iloc[-1] - current_start, 'num_points': len(final_segment)})
    
    time_segments = []
    for segment in segments:
        seg_data = df[(df['timestamp'] >= segment['start']) & (df['timestamp'] <= segment['end'])].copy()
        time_segments.append({'start': segment['start'], 'end': segment['end'], 'data': seg_data})
    
    return real_data, time_segments


def process_ekf_segments(time_segments, final_params, dt_virt=0.5):
    segment_trajectories = []
    
    for time_segment in time_segments:
        run_data = time_segment['data'].to_dict(orient='records')
        if len(run_data) < 2:
            continue
            
        timestamps = [entry['timestamp'] for entry in run_data]
        virtual_ts = np.arange(min(timestamps), max(timestamps), dt_virt).tolist()
        
        try:
            s_filt, P_filt, s_hat, P, _ = ekf_forward(run_data, timestamps, virtual_ts, final_params)
            s_smooth, P_smooth = smoother(s_filt, P_filt, s_hat, P, timestamps, virtual_ts)
            
            segment_trajectories.append({
                'start_time': min(timestamps),
                'end_time': max(timestamps),
                'virtual_ts': virtual_ts,
                'real_ts': timestamps,
                'all_ts': np.sort(virtual_ts + timestamps),
                'trajectory': s_smooth,
                'covariances': P_smooth,
                'run_data': run_data
            })
        except Exception as e:
            print(f"Error processing segment: {e}")
            continue
    
    return segment_trajectories


def extract_observations(real_data):
    left_x, left_y = [], []
    right_x, right_y = [], []
    timestamps = []
    
    for entry in real_data:
        if 'left' in entry:
            left_x.append(entry['left'][0])
            left_y.append(entry['left'][1])
            timestamps.append(entry['timestamp'])
        if 'right' in entry:
            right_x.append(entry['right'][0])
            right_y.append(entry['right'][1])
    
    return (left_x, left_y), (right_x, right_y), timestamps


def get_current_observations(real_data, current_time):
    """Get observations up to current time with their timestamps"""
    current_entries = []
    
    for entry in real_data:
        if entry['timestamp'] <= current_time:
            current_entries.append(entry)
        else:
            break
    
    return current_entries


def get_most_recent_positions(current_entries):
    """Extract the most recent left and right positions from current entries"""
    most_recent_left = None
    most_recent_right = None
    most_recent_left_time = -1
    most_recent_right_time = -1
    
    for entry in current_entries:
        if 'left' in entry and entry['timestamp'] > most_recent_left_time:
            most_recent_left = entry['left']
            most_recent_left_time = entry['timestamp']
        if 'right' in entry and entry['timestamp'] > most_recent_right_time:
            most_recent_right = entry['right']
            most_recent_right_time = entry['timestamp']
    
    return most_recent_left, most_recent_right


def get_trajectory_for_time(segment_trajectories, current_time):
    trajectory_data = []
    current_position = None
    current_covariance = None
    
    for segment in segment_trajectories:
        if segment['start_time'] > current_time:
            continue
            
        segment_current_time = min(current_time, segment['end_time'])
        
        # Find real timestamps that are valid for the current segment
        valid_real_indices = [i for i, t in enumerate(segment['real_ts']) if t <= segment_current_time]
        
        if valid_real_indices:
            # Get trajectory points corresponding to real timestamps
            trajectory_x = []
            trajectory_y = []
            trajectory_covariances = []
            
            for real_idx in valid_real_indices:
                # Find the closest virtual timestamp index for interpolation
                real_t = segment['real_ts'][real_idx]
                
                # Find the closest timestamp
                allts_idx = np.argmin(np.abs(np.array(segment['all_ts']) - real_t))
                
                if allts_idx is not None and allts_idx < len(segment['trajectory']):
                    trajectory_x.append(segment['trajectory'][allts_idx][0])
                    trajectory_y.append(segment['trajectory'][allts_idx][1])
                    if allts_idx < len(segment['covariances']):
                        trajectory_covariances.append(segment['covariances'][allts_idx])
            
            if trajectory_x and trajectory_y:
                trajectory_data.append({
                    'x': trajectory_x,
                    'y': trajectory_y,
                    'covariances': trajectory_covariances
                })
                
                # Store current position uncertainty for later plotting
                if segment['end_time'] >= current_time or segment == segment_trajectories[-1]:
                    current_position = (trajectory_x[-1], trajectory_y[-1])
                    if trajectory_covariances:
                        current_covariance = trajectory_covariances[-1]
    
    return trajectory_data, current_position, current_covariance


def create_uncertainty_ellipse(position, covariance, confidence_level):
    pos_cov = covariance[:2, :2]
    eigenvals, eigenvecs = np.linalg.eigh(pos_cov)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Calculate angle of major axis
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Chi-square critical value for 2 DOF
    chi2_val = chi2.ppf(confidence_level, df=2)
    
    # Calculate ellipse dimensions
    width = 2 * np.sqrt(chi2_val * eigenvals[0])
    height = 2 * np.sqrt(chi2_val * eigenvals[1])
    
    return {
        'center': position,
        'width': width,
        'height': height,
        'angle': angle
    }