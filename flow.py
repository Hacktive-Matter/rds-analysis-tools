import os
from typing import List, Tuple, TypeAlias
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import cv2 as cv # Assuming cv2 is available from the original code environment

# Assuming core, utils, and visualization modules are set up correctly
from core import OpticalFlowConfig, OutputConfig, FlowResults
from utils import vprint
from utils.analysis import group_avg
from utils.setup import setup_csv_writer

FramePair: TypeAlias = Tuple[int, int]
FlowOutput: TypeAlias = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
FlowStats: TypeAlias = Tuple[float, float, float]


def calculate_frame_pairs(num_frames: int, frame_step: int) -> List[FramePair]:
    """Calculate frame pairs for optical flow analysis."""
    end_frame = (int(num_frames / frame_step) - 1) * frame_step
    while end_frame <= 0:
        frame_step = int(np.ceil(frame_step / 5))
        vprint(
            f"Flow field frame step too large for video, dynamically adjusting, new frame step: {frame_step}"
        )
        end_frame = int(num_frames / frame_step) * frame_step

    frame_pairs = []
    for start in range(0, end_frame, frame_step):
        end = start + frame_step
        end = min(end, num_frames - 1)
        frame_pairs.append((start, end))

    if end_frame != num_frames - 1:
        frame_pairs.append((end_frame, num_frames - 1))

    return frame_pairs


def calculate_visualization_frames(
    frame_pairs: List[FramePair], frame_step: int
) -> set:
    """Calculate which frames should have visualizations saved (matching original logic)."""
    if not frame_pairs:
        return set()

    end_point = frame_pairs[-1][0]  # Last start frame
    mid_point_arr = list(range(0, end_point, frame_step))

    if len(mid_point_arr) <= 1:
        return {0} if frame_pairs else set()

    mid_point = mid_point_arr[int((len(mid_point_arr) - 1) / 2)]
    return {0, mid_point, end_point}


def calculate_optical_flow(
    images: np.ndarray,
    frame_pair: FramePair,
    opt_config: OpticalFlowConfig,
) -> Tuple[FlowOutput, Tuple[float, float, float, float]]:
    """Calculate optical flow between two frames."""
    

    start_frame, end_frame = frame_pair

    # Determine frame interval for speed conversion
    if hasattr(opt_config, 'fps') and opt_config.fps > 0:
        frame_int = 1.0 / opt_config.fps
    elif hasattr(opt_config, 'frame_interval_s') and opt_config.frame_interval_s > 0:
        frame_int = opt_config.frame_interval_s
    else:
        frame_int = 1.0
    
    # Convert from pixels/interval to nm/sec
    speed_conversion_factor = opt_config.nm_pixel_ratio / (
        frame_int * (end_frame - start_frame)
    )

    params = (None, 0.5, 3, opt_config.window_size, 3, 5, 1.2, 0)
    flow = cv.calcOpticalFlowFarneback(images[start_frame], images[end_frame], *params)

    flow_reduced = group_avg(flow, opt_config.downsample_factor)
    downU = np.flipud(flow_reduced[:, :, 0])
    downV = -1 * np.flipud(flow_reduced[:, :, 1])

    directions = np.arctan2(downV, downU) # Already in radians, -pi to pi
    speed = np.sqrt(downU**2 + downV**2)
    speed *= speed_conversion_factor  # Convert speed to nm/sec

    theta = np.mean(directions)
    sigma_theta = np.std(directions)
    mean_speed = np.mean(speed)
    std_speed = np.std(speed)  # Add speed standard deviation

    flow: FlowOutput = (downU, downV, directions, speed)
    flow_stats = (theta, sigma_theta, mean_speed, std_speed)

    return flow, flow_stats


def aggregate_flow_stats(
    thetas: List[float],
    sigma_thetas: List[float],
    speeds: List[float],
) -> FlowResults:
    """Aggregate flow statistics."""

    thetas = np.array(thetas)
    sigma_thetas = np.array(sigma_thetas)
    speeds = np.array(speeds)

    # Metric for average direction of flow (-pi, pi) # "Flow Direction"
    mean_theta = np.mean(thetas)
    # Metric for st. dev of flow (-pi, pi) # "Flow Directional Spread"
    mean_sigma_theta = np.mean(sigma_thetas)
    # Metric for avg. speed (units of nm/s) # Average speed
    mean_speed = np.mean(speeds)
    # Calculate delta speed as (v_f - v_i)
    delta_speed = speeds[-1] - speeds[0]

    return FlowResults(mean_speed, delta_speed, mean_theta, mean_sigma_theta)


def save_mean_speed_graph(frame_pairs: List[FramePair], speeds: List[float], speed_stds: List[float], output_dir: str, opt_config: OpticalFlowConfig):
    """Save a graph of mean speed with standard deviation as error bars over time."""
    if not frame_pairs or not speeds:
        return
    
    # Determine x-axis: time or frames
    use_time_axis = False
    frame_interval_s = 1.0
    
    if hasattr(opt_config, 'fps') and opt_config.fps > 0:
        # If fps is available, calculate frame interval from it
        frame_interval_s = 1.0 / opt_config.fps
        use_time_axis = True
    elif hasattr(opt_config, 'frame_interval_s') and opt_config.frame_interval_s > 0:
        # Use existing frame interval
        frame_interval_s = opt_config.frame_interval_s
        use_time_axis = True
    
    if use_time_axis:
        x_values = [pair[0] * frame_interval_s for pair in frame_pairs]
        x_label = 'Time (s)'
        title = 'Mean Speed with Standard Deviation Over Time'
    else:
        x_values = [pair[0] for pair in frame_pairs]
        x_label = 'Frame Number'
        title = 'Mean Speed with Standard Deviation Over Frames'
    
    plt.figure(figsize=(12, 8))
    
    # Plot mean speed with error bars representing standard deviation
    plt.errorbar(x_values, speeds, yerr=speed_stds, fmt='b-o', linewidth=2, markersize=6, 
                 capsize=5, capthick=1.5, elinewidth=1.5, label='Mean Speed ± Std Dev')
    
    plt.xlabel(x_label)
    plt.ylabel('Speed (nm/s)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot in the same directory as other outputs
    graph_filename = os.path.join(output_dir, 'Speed Over Time.png')
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    vprint(f"Speed analysis graph saved to: {graph_filename}")


def save_trajectory_plot(
    frame_pairs: List[FramePair], 
    mean_flow_vectors: List[Tuple[float, float]], 
    output_dir: str, 
    opt_config: OpticalFlowConfig
):
    """
    Save a 2D trajectory plot showing cumulative displacement over time.
    
    Args:
        frame_pairs: List of frame pairs used in analysis
        mean_flow_vectors: List of (mean_u, mean_v) tuples for each frame pair
        output_dir: Directory to save the plot
        opt_config: Optical flow configuration containing conversion parameters
    """
    if not frame_pairs or not mean_flow_vectors:
        vprint("No data available for trajectory plot")
        return
    
    # Determine time parameters
    use_time_axis = False
    frame_interval_s = 1.0
    
    if hasattr(opt_config, 'fps') and opt_config.fps > 0:
        frame_interval_s = 1.0 / opt_config.fps
        use_time_axis = True
    elif hasattr(opt_config, 'frame_interval_s') and opt_config.frame_interval_s > 0:
        frame_interval_s = opt_config.frame_interval_s
        use_time_axis = True
    
    # Calculate cumulative displacement in nanometers
    cumulative_x = [0.0]  # Start at origin
    cumulative_y = [0.0]
    times = [0.0] if use_time_axis else [0]
    
    for i, (frame_pair, (mean_u, mean_v)) in enumerate(zip(frame_pairs, mean_flow_vectors)):
        start_frame, end_frame = frame_pair
        frame_duration = end_frame - start_frame
        
        # Convert flow vectors to displacement in nm
        # mean_u and mean_v are already in pixels per frame interval
        displacement_x = mean_u * opt_config.nm_pixel_ratio
        displacement_y = mean_v * opt_config.nm_pixel_ratio
        
        # Add to cumulative position
        cumulative_x.append(cumulative_x[-1] + displacement_x)
        cumulative_y.append(cumulative_y[-1] + displacement_y)
        
        # Calculate time point
        if use_time_axis:
            time_point = frame_pair[1] * frame_interval_s
        else:
            time_point = frame_pair[1]
        times.append(time_point)
    
    # Create the trajectory plot
    plt.figure(figsize=(12, 10))
    
    # Create a colormap for time progression
    colors = cm.viridis(np.linspace(0, 1, len(cumulative_x)))
    
    # Plot the trajectory line
    plt.plot(cumulative_x, cumulative_y, 'k-', linewidth=1, alpha=0.7, zorder=1)
    
    # Plot points colored by time
    scatter = plt.scatter(cumulative_x, cumulative_y, c=times, cmap='viridis', 
                         s=50, zorder=2, edgecolors='black', linewidth=0.5)
    
    # Mark start and end points
    plt.scatter(cumulative_x[0], cumulative_y[0], c='green', s=100, 
               marker='o', label='Start', zorder=3, edgecolors='black', linewidth=1)
    plt.scatter(cumulative_x[-1], cumulative_y[-1], c='red', s=100, 
               marker='s', label='End', zorder=3, edgecolors='black', linewidth=1)
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter)
    if use_time_axis:
        cbar.set_label('Time (s)', fontsize=12)
        title_suffix = "Over Time"
    else:
        cbar.set_label('Frame Number', fontsize=12)
        title_suffix = "Over Frames"
    
    # Add arrows to show direction at key points
    arrow_indices = np.linspace(0, len(cumulative_x)-2, min(10, len(cumulative_x)-1), dtype=int)
    for i in arrow_indices:
        dx = cumulative_x[i+1] - cumulative_x[i]
        dy = cumulative_y[i+1] - cumulative_y[i]
        if abs(dx) > 1e-10 or abs(dy) > 1e-10:  # Only draw arrow if there's movement
            plt.arrow(cumulative_x[i], cumulative_y[i], dx*0.7, dy*0.7, 
                     head_width=max(abs(dx), abs(dy))*0.1, head_length=max(abs(dx), abs(dy))*0.1,
                     fc='gray', ec='gray', alpha=0.6, zorder=1)
    
    plt.xlabel('Cumulative X Displacement (nm)', fontsize=12)
    plt.ylabel('Cumulative Y Displacement (nm)', fontsize=12)
    plt.title(f'2D Movement Trajectory {title_suffix}', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio for true trajectory shape
    
    # Calculate drift metrics
    total_displacement = np.sqrt(cumulative_x[-1]**2 + cumulative_y[-1]**2)
    path_length = sum(np.sqrt((cumulative_x[i+1]-cumulative_x[i])**2 + 
                             (cumulative_y[i+1]-cumulative_y[i])**2) 
                     for i in range(len(cumulative_x)-1))
    
    # Drift efficiency: 1.0 = perfect straight line drift, 0.0 = pure random walk
    drift_efficiency = total_displacement / path_length if path_length > 0 else 0
    
    # Drift rate (nm per unit time)
    total_time = times[-1] - times[0] if len(times) > 1 and times[-1] > times[0] else 1
    drift_rate = total_displacement / total_time
    
    # Drift direction (angle from positive x-axis)
    drift_angle_rad = np.arctan2(cumulative_y[-1], cumulative_x[-1])
    drift_angle_deg = np.degrees(drift_angle_rad)
    
    # Determine drift interpretation
    if drift_efficiency > 0.7:
        drift_type = "Strong Drift"
    elif drift_efficiency > 0.4:
        drift_type = "Moderate Drift"
    elif drift_efficiency > 0.2:
        drift_type = "Weak Drift"
    else:
        drift_type = "Random Motion"
    
    time_unit = 's' if use_time_axis else 'frames'
    
    plt.text(0.02, 0.98, 
             f'Net Displacement: {total_displacement:.1f} nm\n'
             f'Path Length: {path_length:.1f} nm\n'
             f'Drift Efficiency: {drift_efficiency:.3f}\n'
             f'Drift Type: {drift_type}\n'
             f'Drift Rate: {drift_rate:.1f} nm/{time_unit}\n'
             f'Drift Direction: {drift_angle_deg:.1f}°',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plot_filename = os.path.join(output_dir, '2D Cumulative Trajectory Over Time.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    vprint(f"2D trajectory plot saved to: {plot_filename}")
    vprint(f"Drift Analysis - Net: {total_displacement:.1f} nm, Path: {path_length:.1f} nm, "
           f"Efficiency: {drift_efficiency:.3f} ({drift_type}), Rate: {drift_rate:.1f} nm/{time_unit}")


def write_flow_data(csvwriter, flow: FlowOutput, frame_pair: Tuple[int, int]):
    """Write flow field data to CSV."""
    if not csvwriter:
        return

    start_frame, end_frame = frame_pair
    downU, downV, _, _ = flow

    csvwriter.writerow([f"Flow Field ({start_frame}-{end_frame})"])
    csvwriter.writerow(["X-Direction"])
    csvwriter.writerows(downU)
    csvwriter.writerow(["Y-Direction"])
    csvwriter.writerows(downV)

def save_speed_distribution_plot(all_speeds: List[np.ndarray], output_dir: str, title_suffix: str = ""):
    """
    Saves a histogram of all collected speeds.
    all_speeds: A list of numpy arrays, where each array contains speeds from a frame pair.
    """
    if not all_speeds:
        return

    # Concatenate all speed arrays into a single 1D array
    combined_speeds = np.concatenate(all_speeds).flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(combined_speeds, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Speed (nm/s)')
    plt.ylabel('Probability Density')
    plt.title(f'Speed Distribution {title_suffix}')
    plt.grid(True, alpha=0.3)

    plot_filename = os.path.join(output_dir, f'speed_distribution{title_suffix.replace(" ", "_")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    vprint(f"Speed distribution plot saved to: {plot_filename}")

def save_speed_direction_2d_histogram(
    all_speeds: List[np.ndarray], all_directions: List[np.ndarray], output_dir: str, title_suffix: str = ""
):
    """
    Saves a 2D histogram (heatmap) of speed vs. direction.
    all_speeds: A list of numpy arrays, where each array contains speeds from a frame pair.
    all_directions: A list of numpy arrays, where each array contains directions from a frame pair.
    """
    if not all_speeds or not all_directions:
        return

    # Concatenate all speed and direction arrays
    combined_speeds = np.concatenate(all_speeds).flatten()
    combined_directions = np.concatenate(all_directions).flatten()

    # Ensure directions are in a consistent range, e.g., 0 to 2*pi
    # np.arctan2 returns angles in radians in the range [-pi, pi].
    # We can convert them to [0, 2*pi] for better visualization if desired.
    combined_directions = np.where(combined_directions < 0, combined_directions + 2 * np.pi, combined_directions)

    plt.figure(figsize=(10, 8))
    
    # Create the 2D histogram
    # bins for directions from 0 to 2*pi, bins for speed from 0 to max_speed
    # You might want to adjust the number of bins based on your data and desired resolution
    # vmin and vmax can be adjusted to control the color mapping for density
    
    hist, xedges, yedges, im = plt.hist2d(
        combined_directions, combined_speeds,
        bins=[50, 50], # 50 bins for direction, 50 for speed
        density=True,  # Normalize to show probability density
        cmap='viridis', # Colormap (e.g., 'viridis', 'hot', 'jet')
        cmin=0.0001 # Minimum count for a bin to be colored, helps with sparse data
    )
    
    plt.colorbar(label='Probability Density')
    plt.xlabel('Direction (radians)')
    plt.ylabel('Speed (nm/s)')
    plt.title(f'Speed vs. Direction Distribution {title_suffix}')
    
    # Optionally, set x-ticks for direction to be more readable (e.g., pi/2, pi, etc.)
    # You can customize these based on your preference
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])
    
    plot_filename = os.path.join(output_dir, f'Speed Direction Cumulative Distribution{title_suffix.replace(" ", "_")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    vprint(f"Speed-Direction 2D histogram saved to: {plot_filename}")


def save_static_speed_direction_windows(
    all_speeds: List[np.ndarray],
    all_directions: List[np.ndarray],
    frame_pairs: List[FramePair],
    output_dir: str,
    opt_config: OpticalFlowConfig,
    num_windows: int = 5,
    title_suffix: str = ""
):
    """
    Create 5 static time window plots on one image showing speed vs direction distribution over time.
    
    Args:
        all_speeds: List of numpy arrays, each containing speeds from a frame pair
        all_directions: List of numpy arrays, each containing directions from a frame pair
        frame_pairs: List of frame pairs corresponding to the speed/direction data
        output_dir: Directory to save the plot
        opt_config: Configuration object containing timing information
        num_windows: Number of time windows to create (default 5)
        title_suffix: Additional text for the title
    """
    if not all_speeds or not all_directions or len(all_speeds) < num_windows:
        vprint(f"Insufficient data for {num_windows} time windows (need at least {num_windows} frame pairs)")
        return

    # Determine time parameters
    use_time_axis = False
    frame_interval_s = 1.0
    
    if hasattr(opt_config, 'fps') and opt_config.fps > 0:
        frame_interval_s = 1.0 / opt_config.fps
        use_time_axis = True
    elif hasattr(opt_config, 'frame_interval_s') and opt_config.frame_interval_s > 0:
        frame_interval_s = opt_config.frame_interval_s
        use_time_axis = True

    # Calculate time points for each frame pair
    time_points = []
    for frame_pair in frame_pairs:
        if use_time_axis:
            time_point = frame_pair[0] * frame_interval_s
        else:
            time_point = frame_pair[0]
        time_points.append(time_point)

    # Prepare all directions to be in [0, 2π] range
    all_directions_normalized = []
    for directions in all_directions:
        directions_norm = np.where(directions < 0, directions + 2 * np.pi, directions)
        all_directions_normalized.append(directions_norm)

    # Determine global ranges for consistent binning
    all_speeds_flat = np.concatenate(all_speeds).flatten()
    all_directions_flat = np.concatenate(all_directions_normalized).flatten()
    
    speed_min, speed_max = 0, np.percentile(all_speeds_flat, 99)  # Use 99th percentile to avoid outliers
    direction_bins = np.linspace(0, 2*np.pi, 51)  # 50 bins for direction
    speed_bins = np.linspace(speed_min, speed_max, 51)  # 50 bins for speed
    
    # Calculate window size and indices
    total_pairs = len(all_speeds)
    window_size = max(1, total_pairs // num_windows)  # Size of each window
    
    # Calculate window indices
    window_indices = []
    for i in range(num_windows):
        start_idx = i * window_size
        if i == num_windows - 1:  # Last window gets remaining data
            end_idx = total_pairs
        else:
            end_idx = min((i + 1) * window_size, total_pairs)
        
        if start_idx < total_pairs:
            window_indices.append((start_idx, end_idx))

    # Calculate global max density for consistent color scaling
    global_max_density = 0
    for start_idx, end_idx in window_indices:
        window_speeds = np.concatenate(all_speeds[start_idx:end_idx]).flatten()
        window_directions = np.concatenate(all_directions_normalized[start_idx:end_idx]).flatten()
        hist, _, _ = np.histogram2d(window_directions, window_speeds, 
                                   bins=[direction_bins, speed_bins], density=True)
        global_max_density = max(global_max_density, np.max(hist))

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(window_indices), figsize=(4*len(window_indices), 6))
    
    # Handle case where there's only one subplot
    if len(window_indices) == 1:
        axes = [axes]
    
    time_unit = 's' if use_time_axis else 'frames'
    
    for i, (ax, (start_idx, end_idx)) in enumerate(zip(axes, window_indices)):
        # Get data for current window
        window_speeds = np.concatenate(all_speeds[start_idx:end_idx]).flatten()
        window_directions = np.concatenate(all_directions_normalized[start_idx:end_idx]).flatten()
        
        # Calculate histogram
        hist, _, _ = np.histogram2d(window_directions, window_speeds, 
                                   bins=[direction_bins, speed_bins], density=True)
        
        # Create the plot
        im = ax.imshow(hist.T, extent=[0, 2*np.pi, speed_min, speed_max], 
                      origin='lower', aspect='auto', cmap='viridis', 
                      vmin=0, vmax=global_max_density)
        
        # Set up the subplot
        ax.set_xlabel('Direction (radians)', fontsize=10)
        if i == 0:  # Only label y-axis for leftmost plot
            ax.set_ylabel('Speed (nm/s)', fontsize=10)
        
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Calculate time window information
        start_time = time_points[start_idx]
        end_time = time_points[end_idx-1] if end_idx > 0 else time_points[start_idx]
        start_frame = frame_pairs[start_idx][0]
        end_frame = frame_pairs[end_idx-1][1] if end_idx > 0 else frame_pairs[start_idx][1]
        
        if use_time_axis:
            window_title = f'Window {i+1}\n{start_time:.2f}-{end_time:.2f} {time_unit}'
        else:
            window_title = f'Window {i+1}\nFrames {start_frame}-{end_frame}'
        
        ax.set_title(window_title, fontsize=10)
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability Density', fontsize=12)
    
    # Set overall title
    fig.suptitle(f'Speed vs. Direction Distribution - 5 Time Windows {title_suffix}', fontsize=14)
    
    # Adjust spacing
    fig.subplots_adjust(wspace=0.3, top=0.85)
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f'Speed_Direction_5_Time_Windows{title_suffix.replace(" ", "_")}.png')
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    vprint(f"5 Time Window speed-direction plots saved to: {plot_filename}")


def save_animated_speed_direction_histogram(
    all_speeds: List[np.ndarray], 
    all_directions: List[np.ndarray], 
    frame_pairs: List[FramePair],
    output_dir: str, 
    opt_config: OpticalFlowConfig,
    window_size: int = 5,
    title_suffix: str = ""
):
    """
    Creates an animated 2D histogram showing speed vs direction evolution over time.
    
    Args:
        all_speeds: List of numpy arrays, each containing speeds from a frame pair
        all_directions: List of numpy arrays, each containing directions from a frame pair
        frame_pairs: List of frame pairs corresponding to the speed/direction data
        output_dir: Directory to save the animation
        opt_config: Configuration object containing timing information
        window_size: Number of frame pairs to include in each animation frame (sliding window)
        title_suffix: Additional text for the title
    """
    if not all_speeds or not all_directions or len(all_speeds) < window_size:
        vprint(f"Insufficient data for animated histogram (need at least {window_size} frame pairs)")
        return

    # Determine time parameters
    use_time_axis = False
    frame_interval_s = 1.0
    
    if hasattr(opt_config, 'fps') and opt_config.fps > 0:
        frame_interval_s = 1.0 / opt_config.fps
        use_time_axis = True
    elif hasattr(opt_config, 'frame_interval_s') and opt_config.frame_interval_s > 0:
        frame_interval_s = opt_config.frame_interval_s
        use_time_axis = True

    # Calculate time points for each frame pair
    time_points = []
    for frame_pair in frame_pairs:
        if use_time_axis:
            time_point = frame_pair[0] * frame_interval_s
        else:
            time_point = frame_pair[0]
        time_points.append(time_point)

    # Prepare all directions to be in [0, 2π] range
    all_directions_normalized = []
    for directions in all_directions:
        directions_norm = np.where(directions < 0, directions + 2 * np.pi, directions)
        all_directions_normalized.append(directions_norm)

    # Determine global ranges for consistent binning
    all_speeds_flat = np.concatenate(all_speeds).flatten()
    all_directions_flat = np.concatenate(all_directions_normalized).flatten()
    
    speed_min, speed_max = 0, np.percentile(all_speeds_flat, 99)  # Use 99th percentile to avoid outliers
    direction_bins = np.linspace(0, 2*np.pi, 51)  # 50 bins for direction
    speed_bins = np.linspace(speed_min, speed_max, 51)  # 50 bins for speed
    
    # Calculate global max density for consistent color scaling
    global_max_density = 0
    for i in range(len(all_speeds) - window_size + 1):
        window_speeds = np.concatenate(all_speeds[i:i+window_size]).flatten()
        window_directions = np.concatenate(all_directions_normalized[i:i+window_size]).flatten()
        hist, _, _ = np.histogram2d(window_directions, window_speeds, 
                                   bins=[direction_bins, speed_bins], density=True)
        global_max_density = max(global_max_density, np.max(hist))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create initial empty histogram
    hist_initial = np.zeros((len(direction_bins)-1, len(speed_bins)-1))
    im = ax.imshow(hist_initial.T, extent=[0, 2*np.pi, speed_min, speed_max], 
                   origin='lower', aspect='auto', cmap='viridis', 
                   vmin=0, vmax=global_max_density)
    
    # Set up the plot
    ax.set_xlabel('Direction (radians)', fontsize=12)
    ax.set_ylabel('Speed (nm/s)', fontsize=12)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability Density', fontsize=12)
    
    # Title that will be updated
    time_unit = 's' if use_time_axis else 'frames'
    title = ax.set_title(f'Speed vs. Direction Distribution Over Time {title_suffix}', fontsize=14)
    
    # Text for time window info
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def animate(frame_idx):
        """Animation function called for each frame."""
        # Calculate window indices
        start_idx = frame_idx
        end_idx = frame_idx + window_size
        
        # Get data for current window
        window_speeds = np.concatenate(all_speeds[start_idx:end_idx]).flatten()
        window_directions = np.concatenate(all_directions_normalized[start_idx:end_idx]).flatten()
        
        # Calculate histogram
        hist, _, _ = np.histogram2d(window_directions, window_speeds, 
                                   bins=[direction_bins, speed_bins], density=True)
        
        # Update the image
        im.set_array(hist.T)
        
        # Update time window information
        start_time = time_points[start_idx]
        end_time = time_points[end_idx-1]
        start_frame = frame_pairs[start_idx][0]
        end_frame = frame_pairs[end_idx-1][1]
        
        if use_time_axis:
            time_info = f'Time Window: {start_time:.2f} - {end_time:.2f} s\nFrames: {start_frame} - {end_frame}'
        else:
            time_info = f'Frame Window: {start_frame} - {end_frame}'
        
        time_text.set_text(time_info)
        
        return [im, time_text]

    # Create animation
    num_frames = len(all_speeds) - window_size + 1
    if num_frames <= 0:
        vprint("Not enough frames for animation window")
        plt.close(fig)
        return
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                   interval=500, blit=True, repeat=True)
    
    # Save as GIF
    gif_filename = os.path.join(output_dir, f'Animated_Speed_Direction_Distribution{title_suffix.replace(" ", "_")}.gif')
    try:
        anim.save(gif_filename, writer='pillow', fps=2, dpi=150)
        vprint(f"Animated speed-direction histogram saved to: {gif_filename}")
    except Exception as e:
        vprint(f"Failed to save GIF animation: {e}")
        try:
            # Fallback: try to save as MP4
            mp4_filename = os.path.join(output_dir, f'Animated_Speed_Direction_Distribution{title_suffix.replace(" ", "_")}.mp4')
            anim.save(mp4_filename, writer='ffmpeg', fps=2, bitrate=1800)
            vprint(f"Animated speed-direction histogram saved as MP4 to: {mp4_filename}")
        except Exception as e2:
            vprint(f"Failed to save MP4 animation: {e2}")
            # Save static frames as fallback
            save_static_frames_fallback(fig, animate, num_frames, output_dir, title_suffix)
    
    plt.close(fig)


def save_static_frames_fallback(fig, animate_func, num_frames, output_dir, title_suffix):
    """Fallback function to save individual frames if animation fails."""
    vprint("Animation save failed, saving individual frames as fallback...")
    frames_dir = os.path.join(output_dir, f'Speed_Direction_Frames{title_suffix.replace(" ", "_")}')
    os.makedirs(frames_dir, exist_ok=True)
    
    for i in range(0, num_frames, max(1, num_frames//10)):  # Save every 10th frame or so
        animate_func(i)
        frame_filename = os.path.join(frames_dir, f'frame_{i:03d}.png')
        fig.savefig(frame_filename, dpi=150, bbox_inches='tight')
    
    vprint(f"Static frames saved to: {frames_dir}")


def analyze_flow(
    file: np.ndarray,
    name: str,
    channel: int,
    opt_config: OpticalFlowConfig,
    out_config: OutputConfig,
) -> FlowResults:
    vprint("Beginning Flow Analysis...")

    images = file[:, :, :, channel]
    num_frames = len(images)

    if (images == 0).all():
        return FlowResults()

    csvwriter, myfile = None, None
    if out_config.save_intermediates:
        filename = os.path.join(name, "OpticalFlow.csv")
        csvwriter, myfile = setup_csv_writer(filename)

    thetas, sigma_thetas, speeds, speed_stds = [], [], [], []
    all_individual_speeds = []
    all_individual_directions = [] # List to store all individual direction arrays
    mean_flow_vectors = []  # New list to store mean flow vectors for trajectory
    frame_step = opt_config.frame_step
    frame_pairs = calculate_frame_pairs(num_frames, frame_step)

    # Determine which frames to save visualizations for
    save_frames = set()
    if out_config.save_graphs:
        save_frames = calculate_visualization_frames(frame_pairs, frame_step)

    for frame_pair in frame_pairs:
        start_frame, _ = frame_pair

        flow, flow_stats = calculate_optical_flow(
            images,
            frame_pair,
            opt_config,
        )

        # Extract flow components
        downU, downV, directions, speed = flow
        
        # Store mean flow vectors for trajectory calculation
        mean_u = np.mean(downU)
        mean_v = np.mean(downV)
        mean_flow_vectors.append((mean_u, mean_v))

        # Append the individual speed and direction arrays from this frame pair
        all_individual_speeds.append(speed) 
        all_individual_directions.append(directions)

        # Save visualization for key frames
        if start_frame in save_frames:
            from visualization import save_flow_visualization

            save_flow_visualization(
                flow, start_frame, name, opt_config.downsample_factor
            )

        if csvwriter:
            write_flow_data(csvwriter, flow, frame_pair)

        theta, sigma_theta, mean_speed, std_speed = flow_stats
        thetas.append(theta)
        sigma_thetas.append(sigma_theta)
        speeds.append(mean_speed)
        speed_stds.append(std_speed)

    # Save all plots if graphs are enabled
    if out_config.save_graphs and speeds:
        save_mean_speed_graph(frame_pairs, speeds, speed_stds, name, opt_config)
        save_speed_distribution_plot(all_individual_speeds, name, "Overall")
        save_speed_direction_2d_histogram(all_individual_speeds, all_individual_directions, name, "Overall")
        save_trajectory_plot(frame_pairs, mean_flow_vectors, name, opt_config)

        #Create 5 static time windows
        save_static_speed_direction_windows(
            all_individual_speeds, 
            all_individual_directions, 
            frame_pairs, 
            name, 
            opt_config, 
            num_windows=5,  # Creates exactly 5 time windows
            title_suffix="Overall")

    if myfile:
        myfile.close()

    return aggregate_flow_stats(thetas, sigma_thetas, speeds)
