import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from data_processing_14_model import (
    prepare_real_data, run_ekf_on_segments, get_observations,
    interpolate_trajectory, build_uncertainty_ellipse
)


def animate_ekf(subject_data, final_params, dt_virt=0.5, gif_file='ekf_animation.gif',
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
        ax.set_title(f"EKF Trajectory @ {t:.1f}s")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True)

        (lx, ly), (rx, ry) = get_observations(real_data, t)
        ax.scatter(lx, ly, color='blue', alpha=0.3, s=10, label='Left Obs')
        ax.scatter(rx, ry, color='red', alpha=0.3, s=10, label='Right Obs')

        trajs, pos, cov = interpolate_trajectory(trajectories, t)
        for traj in trajs:
            ax.plot(traj['x'], traj['y'], 'g-', lw=2, alpha=0.8)

        if pos and cov is not None:
            for i, level in enumerate(confidence_levels):
                ellipse_cfg = build_uncertainty_ellipse(pos, cov, level)
                ellipse = Ellipse(
                    xy=ellipse_cfg['center'], width=ellipse_cfg['width'],
                    height=ellipse_cfg['height'], angle=ellipse_cfg['angle'],
                    facecolor=colors[i % len(colors)], edgecolor='black',
                    alpha=0.4, lw=1.0
                )
                ax.add_patch(ellipse)
            ax.scatter(*pos, c='green', s=150, edgecolors='black', lw=2, label='Est. Pos')

        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(animation_ts), interval=100)
    ani.save(gif_file, writer='pillow', fps=10)
    print(f"Saved animation to {gif_file}")
    return ani