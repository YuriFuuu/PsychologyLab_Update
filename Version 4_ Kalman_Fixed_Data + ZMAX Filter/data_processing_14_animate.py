import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from data_processing_14_model import (
    prepare_real_data, run_ekf_on_segments, get_observations,
    interpolate_trajectory, build_uncertainty_ellipse
)
from data_processing_14_ekf import load_data


def animate_ekf(subject_data, final_params, dt_virt=0.5, gif_file='ekf_animation_0807.gif',
                time_step=5, xlim=(0, 15), ylim=(0, 9), confidence_levels=[0.68, 0.95]):
    
    real_data, segments = prepare_real_data(subject_data, dt_virt)
    trajectories = run_ekf_on_segments(segments, final_params, dt_virt)
    
    all_ts = [d['timestamp'] for d in real_data]
    animation_ts = np.arange(min(all_ts), max(all_ts) + time_step, time_step)
    
    fig, ax = plt.subplots(figsize=(15, 9))
    colors = ['lightblue', 'lightcoral']
    
    def update(frame):
        ax.clear()
        t = animation_ts[frame]
        ax.set_title(f"EKF Trajectory @ {t:.1f}s", fontsize=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        (lx, ly), (rx, ry) = get_observations(real_data, t)
        if lx:
            ax.scatter(lx, ly, color='blue', alpha=0.3, s=10, label='Left Obs')
        if rx:
            ax.scatter(rx, ry, color='red', alpha=0.3, s=10, label='Right Obs')
        
        trajs, pos, cov = interpolate_trajectory(trajectories, t)
        for traj in trajs:
            if traj['x'] and traj['y']:
                ax.plot(traj['x'], traj['y'], 'g-', lw=2, alpha=0.8, label='EKF Trajectory')
        
        if pos and cov is not None:
            for i, level in enumerate(confidence_levels):
                try:
                    ellipse_cfg = build_uncertainty_ellipse(pos, cov, level)
                    ellipse = Ellipse(
                        xy=ellipse_cfg['center'], 
                        width=ellipse_cfg['width'],
                        height=ellipse_cfg['height'], 
                        angle=ellipse_cfg['angle'],
                        facecolor=colors[i % len(colors)], 
                        edgecolor='black',
                        alpha=0.4, 
                        lw=1.0,
                        label=f'{int(level*100)}% Confidence' if i == 0 else None
                    )
                    ax.add_patch(ellipse)
                except Exception as e:
                    print(f"Warning: Could not create uncertainty ellipse at time {t:.1f}: {e}")
            
            ax.scatter(*pos, c='green', s=150, edgecolors='black', lw=2, 
                      label='Est. Position', zorder=10)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ani = animation.FuncAnimation(fig, update, frames=len(animation_ts), interval=100, repeat=True)
    try:
        ani.save(gif_file, writer='pillow', fps=10)
        print(f"Saved animation to {gif_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure you have pillow installed: pip install pillow")
    
    return ani


def create_static_plots(subject_data, final_params, dt_virt=0.5):
    real_data, segments = prepare_real_data(subject_data, dt_virt)
    trajectories = run_ekf_on_segments(segments, final_params, dt_virt)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Full trajectory with observations
    (lx, ly), (rx, ry) = get_observations(real_data)
    ax1.scatter(lx, ly, color='blue', alpha=0.3, s=10, label='Left Observations')
    ax1.scatter(rx, ry, color='red', alpha=0.3, s=10, label='Right Observations')
    
    for i, traj in enumerate(trajectories):
        x_vals = [state[0] for state in traj['trajectory']]
        y_vals = [state[1] for state in traj['trajectory']]
        ax1.plot(x_vals, y_vals, 'g-', lw=2, alpha=0.8, 
                label='EKF Trajectory' if i == 0 else None)
    
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Complete EKF Trajectory')
    ax1.legend()
    
    # Plot 2: Velocity over time
    for i, traj in enumerate(trajectories):
        times = traj['all_ts']
        velocities = [np.sqrt(state[3]**2 + state[4]**2) for state in traj['trajectory']]
        ax2.plot(times, velocities, 'b-', lw=2, alpha=0.8, 
                label='Speed' if i == 0 else None)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (units/s)')
    ax2.set_title('Estimated Speed Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('ekf_static_plots.png', dpi=300, bbox_inches='tight')
    print("Saved static plots to ekf_static_plots.png")
    plt.show()


def main():
    try:
        df = load_data(date_index=0)
        final_params = [0.1, 0.01, 0.5, 0.23]
        ani = animate_ekf(df, final_params, dt_virt=0.5, time_step=2.0)
        create_static_plots(df, final_params, dt_virt=0.5)
        plt.show()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()