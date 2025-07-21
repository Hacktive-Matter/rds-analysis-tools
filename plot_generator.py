import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# frames: {name: [values_idx, list, overall_list]}
# color_conditions: {condition: color}
# plot_conditions: {idx: condition}

def build_distribution_plots(files: list, sheet_name: str, frames: dict, color_conditions: dict, plot_conditions: dict, frame_markers, figure_path, xlabel = None, ylabel = None, xlim = None, ylim = None, xticks = None):
    
    fig, ax = plt.subplots(2, len(plot_conditions), figsize = (6 * len(plot_conditions), 12))
    
    def normalize_counts(count): return count / count.sum()
    
    def populate_values(values, counts):
        updated = []
        for value, count in zip(values, counts):
            if str(value) != 'nan' and str(count) != 'nan':
                updated.extend([value] * int(count))
        return updated
    
    def update_direction(directions, sheet_name=None):
        if sheet_name != "direction_distribution":
            return directions
        directions = directions - np.pi/2
        return np.where(directions < 0, directions + 2 * np.pi, directions)
    
    for file in files:
        for color in color_conditions:
            if color in file:
                c = color
        for plot in plot_conditions:
            if plot_conditions[plot] in file:
                m = plot
        
        totals = []
        df = pd.read_excel(file, sheet_name, header = [0, 1], index_col = 0)
        for i in range(0, len(np.unique(df.columns.get_level_values(0)))):
            df_cols = pd.Index([f"File {i + 1}:" in file for file in df.columns.get_level_values(0)])
            filtered_df = df.loc[:,df_cols].dropna(how = 'all').loc[:]
            total_index = filtered_df.last_valid_index()
            for frame in frames:
                val_idx = int(frames[frame][0] - 1) if frames[frame][0] > 0 else int(total_index + (2 * frames[frame][0] - 1))
                end_idx = -2# if "optical_flow" in file else 0
                values = filtered_df.loc[val_idx].to_list()[2:end_idx]
                counts = filtered_df.loc[val_idx + 1].to_list()[2:end_idx]
                frames[frame][1].extend(populate_values(values, counts))
            totals.extend(populate_values(filtered_df.loc[total_index-1].to_list()[2:end_idx], filtered_df.loc[total_index].to_list()[2:end_idx]))
            
        for i, frame in enumerate(frames):
            frames[frame][2] = np.unique(frames[frame][1], return_counts = True)
            label = f'{c.capitalize()} - {frame} Frame'
            ax[0, m].scatter(update_direction(frames[frame][2][0], sheet_name), normalize_counts(frames[frame][2][1]), label = label, marker = frame_markers[i], c = color_conditions[c], alpha = 0.4)
            title = f'{plot_conditions[m].capitalize()} - Individual Frames'
            ax[0, m].legend()
            ax[0, m].set_title(title)

        total = np.unique(totals, return_counts = True)
        label = f'{c.capitalize()} - All Frames'
        ax[1, m].scatter(update_direction(total[0], sheet_name), normalize_counts(total[1]), label = label, marker = frame_markers[i + 1], c = color_conditions[c], alpha = 0.4)
        title = f'{plot_conditions[m].capitalize()} - All Frames'
        ax[1, m].legend()
        ax[1, m].set_title(title)
    if xticks:
        plt.setp(ax, xlabel = xlabel, ylabel = ylabel, xlim = xlim, ylim = ylim, xticks = xticks)
    else:
        plt.setp(ax, xlabel = xlabel, ylabel = ylabel, xlim = xlim, ylim = ylim)
    fig.savefig(figure_path)
    plt.clf()
    plt.close('all')
    return

def create_time_plots(files: list, sheet_name: str, metric: str, color_conditions: dict, plot_conditions: dict, figure_path, xlabel = None, ylabel = None, xlim = None, ylim = None, xticks = None, s_metric: str = None):
    fig, ax = plt.subplots(1, len(plot_conditions), figsize = (6 * len(plot_conditions), 6))
    for file in files:
        for color in color_conditions:
            if color in file:
                c = color
        for plot in plot_conditions:
            if plot_conditions[plot] in file:
                m = plot

        df = pd.read_excel(file, sheet_name, header = [0, 1], index_col = 0)
        condition_cols = df.columns.get_level_values(1).isin(["Frames", metric]) if not s_metric else df.columns.get_level_values(1).isin(["Frames", metric, s_metric])
        mean_cols = df.columns.get_level_values(1).isin([metric])
        df['Mean Speed'] = df[mean_cols].mean()
        for i in range(0, len(np.unique(df.columns.get_level_values(0)))):
            df_cols = pd.Index(np.logical_or(np.logical_and([f"File {i + 1}:" in file for file in df.columns.get_level_values(0)], condition_cols)))
            filtered_df = df.loc[:,df_cols].dropna(how = 'all').loc[:]
            filtered_df.columns = filtered_df.columns.get_level_values(1)
            total_index = filtered_df.last_valid_index()
            last_index = total_index if 'intensity_dist' in file else total_index - 1
            if s_metric:
                filtered_df[:last_index:10].plot(x = "Frames", y = metric, yerr = s_metric, ax = ax[m], c = color_conditions[c], alpha = 0.4)
            else:
                filtered_df[:last_index:10].plot(x = "Frames", y = metric, ax = ax[m], c = color_conditions[c], alpha = 0.4)
    if xticks:
        plt.setp(ax, xlabel = xlabel, ylabel = ylabel, xlim = xlim, ylim = ylim, xticks = xticks)
    else:
        plt.setp(ax, xlabel = xlabel, ylabel = ylabel, xlim = xlim, ylim = ylim)
    fig.savefig(figure_path)
    plt.clf()
    plt.close('all')
    return