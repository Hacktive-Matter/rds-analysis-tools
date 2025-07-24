
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def compute_avg_squared_distance(particles):
    particles = np.array(particles)

    x = particles[:, 0]
    y = particles[:, 1]
    frames = particles[:, 4]

    avg_sq_dists = np.zeros(len(particles))  

    unique_frames = np.unique(frames)

    for f in unique_frames:
        indices = np.where(frames == f)[0]
        coords = np.stack((x[indices], y[indices]), axis=1)

        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  
        dists_squared = np.sum(diffs**2, axis=2)  

        np.fill_diagonal(dists_squared, np.nan)

        mean_sq_dist = np.nanmean(dists_squared, axis=1)

        avg_sq_dists[indices] = mean_sq_dist

    return avg_sq_dists

def visualize_area_vs_time(particles, output_dir="area_vs_time_plots", filename_area="area_vs_time.png"):
    """
    Create a scatter plot of particle areas over time, colored by particle label.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import ListedColormap

    os.makedirs(output_dir, exist_ok=True)

    frames = np.array([p[4] for p in particles])
    labels = np.array([p[2] for p in particles])
    areas = np.array([p[3] for p in particles])
    
    colors_rgb = [
        [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120], [44, 160, 44],
        [152, 223, 138], [214, 39, 40], [255, 152, 150], [148, 103, 189], [197, 176, 213],
        [140, 86, 75], [196, 156, 148], [227, 119, 194], [247, 182, 210], [127, 127, 127],
        [199, 199, 199], [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
        [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
        [255, 217, 47], [229, 196, 148], [179, 179, 179],
        [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
        [255, 217, 47], [229, 196, 148], [179, 179, 179], [179, 179, 179],
        [251, 180, 174], [179, 205, 227], [204, 235, 197], [222, 203, 228], [254, 217, 166],
        [255, 255, 204], [229, 216, 189], [253, 218, 236], [242, 242, 242]
    ]
    num_colors = len(colors_rgb)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    colors = []
    for label in labels:
        label_int = int(label)
        color_idx = (label_int - 1) % num_colors
        colors.append(np.array(colors_rgb[color_idx]) / 255.0)  # Normalize to [0,1]

    scatter = plt.scatter(frames, areas, c=colors, s=30, alpha=0.7, edgecolors='none')
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Areas (pixels^2)', fontsize=12)
    plt.title('Areas vs Frame Number (Colored by Particle ID)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename_area), dpi=150)
    plt.close()
    
    print(f"Plot saved to {os.path.join(output_dir, filename_area)}")
     
def visualize_dist_vs_time(particles, output_dir="area_vs_time_plots", filename_dist="avg_sq_dists_vs_time.png"):
    """
    Create a scatter plot of particle avg_sq_dists over time, using the same colors as video labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import ListedColormap

    os.makedirs(output_dir, exist_ok=True)
    
    frames = np.array([p[4] for p in particles])
    labels = np.array([p[2] for p in particles])
    avg_sq_dists = compute_avg_squared_distance(particles)  # Assume this is defined elsewhere
    
    colors_rgb = [
        [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120], [44, 160, 44],
        [152, 223, 138], [214, 39, 40], [255, 152, 150], [148, 103, 189], [197, 176, 213],
        [140, 86, 75], [196, 156, 148], [227, 119, 194], [247, 182, 210], [127, 127, 127],
        [199, 199, 199], [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
        [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
        [255, 217, 47], [229, 196, 148], [179, 179, 179],
        [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
        [255, 217, 47], [229, 196, 148], [179, 179, 179], [179, 179, 179],
        [251, 180, 174], [179, 205, 227], [204, 235, 197], [222, 203, 228], [254, 217, 166],
        [255, 255, 204], [229, 216, 189], [253, 218, 236], [242, 242, 242]
    ]
    num_colors = len(colors_rgb)
    
    plt.figure(figsize=(12, 8))
    
    # Assign colors to each point based on label
    colors = []
    for label in labels:
        label_int = int(label)
        color_idx = (label_int - 1) % num_colors
        colors.append(np.array(colors_rgb[color_idx]) / 255.0)  # Normalize to [0,1]
    
    plt.scatter(frames, avg_sq_dists, c=colors, s=30, alpha=0.7, edgecolors='none')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Particle avg_sq_dists (pixels)', fontsize=12)
    plt.title('Avg_sq_dists vs Frame Number (Colored by Particle ID)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename_dist), dpi=150)
    plt.close()
    
    print(f"Plot saved to {os.path.join(output_dir, filename_dist)}")

def track_particles(video, min_area=5, max_distance=20.0, binary_threshold=0.5, output_dir=None, 
                    regeneration=True):
    """
    Track particles and save labeled frames to a directory, excluding particles touching edges.
    
    Parameters:
    video (np.ndarray): Input video array [frames, height, width] (float64)
    min_area (int): Minimum area threshold to filter small particles
    max_distance (float): Maximum allowed distance between centroids
    binary_threshold (float): Threshold to convert float to binary
    output_dir (str): Directory to save labeled frames (None to disable)
    regeneration (bool): Whether to regenerate particle labels
    
    Returns:
    list: List of particles with [x, y, label, area, frame]
    """
    import cv2
    import numpy as np
    import os
    from scipy.optimize import linear_sum_assignment

    binary_video = ((video > binary_threshold) * 255).astype(np.uint8)
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    all_particles = []
    next_global_label = 1
    active_particles = {}
    
    height, width = binary_video.shape[1], binary_video.shape[2]
    
    frame_end_number = binary_video.shape[0]
    
    #
    #
    #
    #
    #
    #frame number for testing, should delete or modify the next line to control the end of frames to analyze
    #frame_end_number = 30
    for frame_index in range(frame_end_number):
        frame = binary_video[frame_index]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frame, connectivity=8
        )
        
        current_particles = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            
            touches_edge = (x == 0) or (y == 0) or (x + w == width) or (y + h == height)
            if touches_edge:
                continue  
                
            cx, cy = centroids[label]
            current_particles.append({
                'centroid': (cx, cy),
                'area': area,
                'label': label
            })
        
        if frame_index == 0:
            for p in current_particles:
                p['global_label'] = next_global_label
                all_particles.append([
                    p['centroid'][0], p['centroid'][1],
                    next_global_label, p['area'], frame_index
                ])
                active_particles[next_global_label] = p['centroid']
                next_global_label += 1
            continue
        
        if not active_particles or not current_particles:
            active_particles = {}
            continue
        
        prev_labels = list(active_particles.keys())
        prev_centroids = list(active_particles.values())
        curr_centroids = [p['centroid'] for p in current_particles]
        
        cost_matrix = np.zeros((len(prev_centroids), len(curr_centroids)))
        for i, prev_pt in enumerate(prev_centroids):
            for j, curr_pt in enumerate(curr_centroids):
                dist = np.linalg.norm(np.array(prev_pt) - np.array(curr_pt))
                cost_matrix[i, j] = dist if dist <= max_distance else 1e9
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_curr = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e9:
                global_label = prev_labels[r]
                current_particles[c]['global_label'] = global_label
                matched_curr.add(c)
        for j, p in enumerate(current_particles):
            if j not in matched_curr:
                p['global_label'] = next_global_label
                next_global_label += 1
        for p in current_particles:
            all_particles.append([
                p['centroid'][0], p['centroid'][1],
                p['global_label'], p['area'], frame_index
            ])
            active_particles[p['global_label']] = p['centroid']

    if regeneration and all_particles:
        all_particles = np.array(all_particles)
        col3 = all_particles[:, 2]  
        frames = np.unique(all_particles[:, -1])  

        valid_values = []
        unique_values = np.unique(col3)
        for val in unique_values:
            frames_with_val = np.unique(all_particles[all_particles[:, 2] == val][:, -1])
            if set(frames).issubset(set(frames_with_val)):
                valid_values.append(val)

        filtered = all_particles[np.isin(all_particles[:, 2], valid_values)]

        relabel_map = {val: i + 1 for i, val in enumerate(sorted(valid_values))}
        filtered[:, 2] = [relabel_map.get(v, v) for v in filtered[:, 2]]
        all_particles = filtered.tolist()
    
    # Second pass: generate output frames using final all_particles
    if output_dir is not None:
        for frame_index in range(binary_video.shape[0]):
            frame = binary_video[frame_index]
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            frame_particles = [p for p in all_particles if p[4] == frame_index]
            
            for particle in frame_particles:
                x, y, label, area, _ = particle
                cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(vis_frame, str(int(label)), 
                           (int(x)+5, int(y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_index:04d}.png"), vis_frame)
    
    if output_dir is not None:
        # Define the same color palette as in the scatter plot
        colors_rgb = [
            [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120], [44, 160, 44],
            [152, 223, 138], [214, 39, 40], [255, 152, 150], [148, 103, 189], [197, 176, 213],
            [140, 86, 75], [196, 156, 148], [227, 119, 194], [247, 182, 210], [127, 127, 127],
            [199, 199, 199], [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
            [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
            [255, 217, 47], [229, 196, 148], [179, 179, 179],
            [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195], [166, 216, 84],
            [255, 217, 47], [229, 196, 148], [179, 179, 179], [179, 179, 179],
            [251, 180, 174], [179, 205, 227], [204, 235, 197], [222, 203, 228], [254, 217, 166],
            [255, 255, 204], [229, 216, 189], [253, 218, 236], [242, 242, 242]
        ]
        num_colors = len(colors_rgb)

        for frame_index in range(binary_video.shape[0]):
            frame = binary_video[frame_index]
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            frame_particles = [p for p in all_particles if p[4] == frame_index]
            
            for particle in frame_particles:
                x, y, label, area, _ = particle
                label_int = int(label)
                color_idx = (label_int - 1) % num_colors
                r, g, b = colors_rgb[color_idx]
                color_bgr = (b, g, r)  
                
                cv2.circle(vis_frame, (int(x), int(y)), 3, color_bgr, -1)
                
                cv2.putText(vis_frame, str(label_int), 
                           (int(x)+5, int(y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
            
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_index:04d}.png"), vis_frame)
    
    
    return all_particles

#Generate MSD and plotting
from collections import defaultdict

def compute_msd_and_direction(particles):
    """
    Compute MSD and direction of displacement for each particle across frames.

    Parameters:
        particles (list of tuples): Each tuple is (x, y, label, area, frame)
    
    Returns:
        new_data: list of tuples (x, y, label, area, frame, msd, direction)
    """
    label_to_frames = defaultdict(list)
    for x, y, label, area, frame in particles:
        label_to_frames[label].append((frame, x, y, area))

    new_data = []
    for label, frames in label_to_frames.items():
        frames.sort()
        prev_x, prev_y = None, None

        for i, (frame, x, y, area) in enumerate(frames):
            if i == 0:
                msd = 0.0
                direction = 0.0
            else:
                dx = x - prev_x
                dy = y - prev_y
                msd = dx**2 + dy**2  # mean squared displacement between frames
                direction = np.arctan2(dy, dx)  # in radians

            new_data.append((x, y, label, area, frame, msd, direction))
            prev_x, prev_y = x, y

    return new_data


def plot_msd_vs_time(particles, save_folder='output_plots'):
    data = compute_msd_and_direction(particles)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    frames = [d[4] for d in data]
    msds = [d[5] for d in data]

    plt.figure()
    plt.scatter(frames, msds, alpha=0.5, s=10)
    plt.xlabel('Frame')
    plt.ylabel('MSD (pixels²)')
    plt.title('MSD vs Time')
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_folder, 'msd_vs_time.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_msd_distribution(particles, save_folder='output_plots'):
    data = compute_msd_and_direction(particles)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    msds = [d[5] for d in data]

    plt.figure()
    plt.hist(msds, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel('MSD (pixels²)')
    plt.ylabel('Count')
    plt.title('MSD Distribution')
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_folder, 'msd_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_msd_vs_time_by_label(particles, save_folder='output_plots'):
    data = compute_msd_and_direction(particles)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    frames = np.array([d[4] for d in data])
    msds = np.array([d[5] for d in data])
    labels = np.array([d[2] for d in data])

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    base_cmap = plt.cm.get_cmap('tab20', num_labels)
    colors = base_cmap(np.linspace(0, 1, num_labels))
    custom_cmap = ListedColormap(colors)

    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    color_indices = np.array([label_to_color[l] for l in labels])

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(frames, msds, c=color_indices, cmap=custom_cmap, s=30, alpha=0.7, edgecolors='none')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('MSD (pixels²)', fontsize=12)
    plt.title('MSD vs Time (colored by Particle ID)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    cbar = plt.colorbar(scatter, ticks=np.arange(num_labels))
    cbar.ax.set_yticklabels([str(l) for l in unique_labels])
    cbar.set_label('Particle ID')

    # Save
    plt.tight_layout()
    filename = os.path.join(save_folder, 'msd_vs_time_by_label.png')
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Plot saved to {filename}")

#Research on pairs of particles
def plot_particle_pair_distance_and_area(particles, label_1, label_2, output_dir='output_plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    particles = np.array(particles)

    label_1 = int(label_1)
    label_2 = int(label_2)

    part1 = particles[particles[:, 2] == label_1]
    part2 = particles[particles[:, 2] == label_2]

    # Sort by frame number to match over time
    part1 = part1[np.argsort(part1[:, 4])]
    part2 = part2[np.argsort(part2[:, 4])]

    frames1 = part1[:, 4].astype(int)
    frames2 = part2[:, 4].astype(int)
    common_frames = np.intersect1d(frames1, frames2)

    times = []
    distances = []
    areas1 = []
    areas2 = []

    for frame in common_frames:
        p1 = part1[part1[:, 4] == frame][0]
        p2 = part2[part2[:, 4] == frame][0]

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        times.append(frame)
        distances.append(dist)
        areas1.append(float(p1[3]))
        areas2.append(float(p2[3]))

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(times, areas1, 'b-o', label=f'Area (label {label_1})')
    ax1.plot(times, areas2, 'g-o', label=f'Area (label {label_2})')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Particle Area', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(times, distances, 'r--s', label='Distance', linewidth=2)
    ax2.set_ylabel('Distance Between Particles', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title(f'Area and Distance Over Time for Particles {label_1} & {label_2}', fontsize=14)
    plt.tight_layout()

    filename = os.path.join(output_dir, f'distance_area_labels_{label_1}_{label_2}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Plot saved to {filename}") 





def plot_particle_trajectory(particles, particle_label_1, output_folder):
    """
    Plot the trajectory of a specific particle (identified by label) over time using its centroid positions.

    Parameters:
        particles (list of lists or ndarray): Each row = [centroid_x, centroid_y, label, area, frame]
        particle_label_1 (int or float): The label of the particle to track
        output_folder (str): Folder path to save the trajectory plot
    """
    particles = np.array(particles)
    label_col = particles[:, 2].astype(float)
    particle_data = particles[label_col == particle_label_1]

    if particle_data.shape[0] == 0:
        print(f"[Warning] No data found for particle label {particle_label_1}")
        return

    particle_data = particle_data[np.argsort(particle_data[:, 4].astype(int))]

    x = particle_data[:, 0].astype(float)
    y = particle_data[:, 1].astype(float)
    frames = particle_data[:, 4].astype(int)

    # Create plot
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, '-o', label=f'Label {particle_label_1}', markersize=4)
    for xi, yi, fi in zip(x, y, frames):
        plt.text(xi, yi, str(fi), fontsize=7, ha='right', va='bottom')

    plt.title(f'Trajectory of Particle Label {particle_label_1}')
    plt.xlabel('Centroid X')
    plt.ylabel('Centroid Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # Save the figure
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f'trajectory_label_{int(particle_label_1)}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] Trajectory plot for label {particle_label_1} saved to:\n{save_path}")
      

def main_scatter_plotting(video, min_area=5, max_distance=20.0, binary_threshold=0.5, output_dir=None, 
                    regeneration = True, plot_msd = True):
    particles = track_particles(video = video, min_area = min_area, max_distance = max_distance, 
                                binary_threshold = binary_threshold, 
                                output_dir = output_dir, regeneration= regeneration)
    visualize_area_vs_time(particles, output_dir)
    visualize_dist_vs_time(particles, output_dir)
    if plot_msd == True:
        plot_msd_distribution(particles, save_folder = output_dir)
        plot_msd_vs_time_by_label(particles, save_folder = output_dir)
        plot_msd_distribution(particles, save_folder = output_dir)

        


# Example usage
##if __name__ == "__main__":
    # Assuming you have particles from track_particles()
    #particles = track_particles(your_video)
    
    # Generate visualizations
    #visualize_particle_areas(
    #    particles,
    #    output_dir="particle_area_analysis",
    #    per_frame=True
    #)
