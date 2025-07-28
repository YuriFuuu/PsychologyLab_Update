import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from IPython.display import clear_output
import time
from data_processing_13_part3_model import compute_smoothed_trajectory, process_ekf_segments, extract_observations, get_current_observations, get_trajectory_for_time, create_uncertainty_ellipse


# Load the data

file_path = './Synched_Data_GR0_22_DEN_MAXZ1_25/NEWDATA/'
file_date = ['101922', '102122', '111422', '111622', '120522', '120722', '013023', '020123', '031323', '031523', '041723', '041923', '061523']
date = file_date[0]

file_name = f'DAYUBIGR_{date}_GR0_22_DEN_032825_V2392628911.CSV'
full_path = file_path + file_name

raw_data = pd.read_csv(full_path, header=None, names=['SUBJECTID', 'TIME', 'X', 'Y', 'Z'])
clear_data = raw_data.reset_index(drop=True)
clear_data = clear_data[(clear_data["X"] <= 15) & (clear_data["Y"] <= 9) & 
                        (clear_data["X"] >= 0) & (clear_data["Y"] >= 0)].copy()
target_subject_base = "DS_STARFISH_2223_27"
subject_data = clear_data[clear_data['SUBJECTID'].str.startswith(target_subject_base)].copy()
subject_data['side'] = subject_data['SUBJECTID'].str.extract(r'(\d+[LR])$')[0].str[-1].map({'L': 'left', 'R': 'right'})

# The chart of EKF smoothed trajectory with time segments

def ekf_trajectory_animation(subject_data, 
                           final_params=[0.06787, 0.08011, 0.04829, 0.38712, 0.41107], 
                           dt_virt=0.5, 
                           save_gif=False, 
                           gif_filename='ekf_animation.gif', 
                           show_uncertainty=True, 
                           confidence_levels=[0.68, 0.95],
                           time_step=5,
                           xlim=(0, 15),
                           ylim=(0, 9)):

    real_data, time_segments = compute_smoothed_trajectory(subject_data, final_params, dt_virt)   
    segment_trajectories = process_ekf_segments(time_segments, final_params, dt_virt)

    (left_x, left_y), (right_x, right_y) = extract_observations(real_data)
    
    all_timestamps = [entry['timestamp'] for entry in real_data]
    min_time, max_time = min(all_timestamps), max(all_timestamps)
    animation_times = np.arange(min_time, max_time + time_step, time_step)
    
    fig, ax = plt.subplots(figsize=(15, 9))
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    def animate_frame(frame):
        current_time = animation_times[frame]
        ax.clear()
        ax.set_title(f'EKF Trajectory Animation')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        (current_left_x, current_left_y), (current_right_x, current_right_y) = get_current_observations(real_data, current_time)
        
        if current_left_x:
            ax.scatter(current_left_x, current_left_y, color='blue', alpha=0.3, s=10, label='Left observations')
        if current_right_x:
            ax.scatter(current_right_x, current_right_y, color='red', alpha=0.3, s=10, label='Right observations')
        
        trajectory_data, current_position, current_covariance = get_trajectory_for_time(segment_trajectories, current_time)
        
        first_trajectory = True
        for traj in trajectory_data:
            label = 'Smoothed Trajectory' if first_trajectory else None
            first_trajectory = False
            ax.plot(traj['x'], traj['y'], 'g-', alpha=0.8, linewidth=2, label=label)
        
        if show_uncertainty and current_position and current_covariance is not None:
            for j, conf_level in enumerate(confidence_levels):
                ellipse_params = create_uncertainty_ellipse(current_position, current_covariance, conf_level)
                
                ellipse = Ellipse(xy=ellipse_params['center'],
                                width=ellipse_params['width'], 
                                height=ellipse_params['height'],
                                angle=ellipse_params['angle'],
                                facecolor=colors[j % len(colors)], 
                                alpha=0.4, 
                                edgecolor='black', 
                                linewidth=1.0)
                ax.add_patch(ellipse)
        
        if show_uncertainty:
            for j, conf_level in enumerate(confidence_levels):
                ax.scatter([], [], color=colors[j % len(colors)], alpha=0.4, s=100, 
                          label=f'{conf_level*100:.0f}% Confidence')
        
        if current_position:
            ax.scatter(current_position[0], current_position[1], c='green', s=150, 
                      edgecolors='black', linewidth=2, marker='o', zorder=5, 
                      label='Current Position')
        
        ax.legend()
    
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(animation_times), interval=50, repeat=True, blit=False)
    ani.save(gif_filename, writer='pillow', fps=10)
    
    return ani


ani = ekf_trajectory_animation(subject_data, confidence_levels=[0.68, 0.95], time_step=5)

# plt.show()