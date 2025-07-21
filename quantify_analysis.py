import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kurtosis, mode
import numpy as np
import csv, os, yaml, sys, re, nd2, cv2
from itertools import pairwise, zip_longest
from skimage.measure import label, regionprops
from import_rds import read_flow_fields, read_bin_frames, read_intensity_dist
import pandas as pd
import xarray as xr
import awkward as ak
from matplotlib import colors
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
from analysis_functions import med_skew, mode_skew, pad_for_df, get_num_frames
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree 


import warnings
warnings.filterwarnings("ignore")

def get_island_orientation(frame):
    image = frame.astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    anisotropy_factors = []
    angles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        angles.append(angle * np.pi / 180)
        anisotropy_factors.append(MA/ma)
    
    return angles, anisotropy_factors

def get_island_area_stats(frame):
    labeled = label(frame, connectivity = 2)
    regions = regionprops(labeled)
    regions = sorted(regions, key = lambda r: r.area, reverse = True)
    areas = np.array([region.area/(frame.shape[0] * frame.shape[1]) for region in regions if region.area > 1])
    return areas

def get_island_distances(frame):
    idxs = []
    ids = []
    image = frame.astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counter = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2:
            continue
        ids.append(counter)
        idxs.append(contour[:,0])
        counter += 1

    trees = [cKDTree(pts) for pts in idxs]

    pdists = {}
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            # query all pts in island i against island j’s tree
            dists, _ = trees[j].query(idxs[i], k=1)
            pdists[(ids[i], ids[j])] = float(dists.min())
            
    return pdists

def flow(path, size = 1, size_units = 'um', frame_interval = 1, time_units = 's', file_location = None):
    def get_flow_field_information(frame):

        def divergence_curl_2d(flow):
            flow = np.swapaxes(flow, 0, 1)
            f = [flow[:,:,0], flow[:,:,1]]
            theta = np.arctan2(flow[:,:,1], flow[:,:,0])
            angles = [np.cos(theta), np.sin(theta)]
            divergence = np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(len(f))])
            radial_divergence = np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) * angles[i] for i in range(len(f))])
            curl = np.ufunc.reduce(np.subtract, [np.gradient(f[len(f)-1-i], axis=i) for i in range(len(f))])
            return divergence, curl, radial_divergence
        
        divergence, curl, radial = divergence_curl_2d(frame)
        speeds = np.hypot(frame[:,:,0], frame[:,:,1])
        directions = np.arctan2(frame[:,:,0], frame[:,:,1])

        return speeds, directions, divergence, curl, radial
    
    frames, frame_idx = read_flow_fields(path)
    divergence_list = []
    speeds_list = []
    directions_list = []
    
    total_speeds = []
    total_divergences = []
    total_directions = []

    for i in range(len(frames)):
        frame_stride = frame_idx[i][1] - frame_idx[i][0]
        frame_stride = 1
        speed, direction, div, _, _ = get_flow_field_information(frames[i])

        speeds_list.append(np.unique(np.round(speed.flatten() * size * 2 / (frame_stride * frame_interval), 1)/2, return_counts = True))
        directions_list.append(np.unique(np.round(direction.flatten(), 1), return_counts = True))
        divergence_list.append(np.unique(np.round(div.flatten(), 2), return_counts = True))

        total_speeds.extend(np.round(speed.flatten() * size * 2 / (frame_stride * frame_interval), 1)/2)
        total_directions.extend(np.round(direction.flatten(), 1))
        total_divergences.extend(np.round(div.flatten(), 2))

    speeds_list.append(np.unique(total_speeds, return_counts = True))
    directions_list.append(np.unique(total_directions, return_counts = True))
    divergence_list.append(np.unique(total_divergences, return_counts=True))

    frame_indices = pd.MultiIndex.from_product([sorted([frame_idx[i][0] for i in range(len(frame_idx))]) + ['Total'], ['', 'Count']])
    # print(frame_indices)
    num_speeds = len(speeds_list[-1][0])
    num_directions = len(directions_list[-1][0])
    num_divergences = len(divergence_list[-1][0])
    speeds_list, speed_counts = pad_for_df(speeds_list, True)
    directions_list, direction_counts = pad_for_df(directions_list, True)
    divergence_list, divergence_counts = pad_for_df(divergence_list, True)

    combined_speeds = []
    combined_directions = []
    combined_divergences = []

    for s, sc, d, dc, div, divc in zip(speeds_list, speed_counts, directions_list, direction_counts, divergence_list, divergence_counts):
        combined_speeds.extend([s, sc])
        combined_directions.extend([d, dc])
        combined_divergences.extend([div, divc])

    summary_df = pd.DataFrame({
        "Frames" : sorted([frame_idx[i][0] for i in range(len(frame_idx))]) + ['Total'],
        "Mean Speed" : [np.nanmean(s[:-2]) for s in speeds_list],
        'Speed Standard Deviation' : [np.nanstd(s[:-2]) for s in speeds_list],
        "Mean Direction" : [np.nanmean(d[:-2]) for d in directions_list],
        'Direction Standard Deviation' : [np.nanstd(d[:-2]) for d in directions_list],
        "Mean Divergence": [np.nanmean(div[:-2]) for div in divergence_list]
    })
        
    speed_iter = pd.MultiIndex.from_product([[f"Speed ({size_units}/{time_units})"], [_ for _ in range(num_speeds)] + ['Mean Speed', 'Speed Standard Deviation']])
    dir_iter = pd.MultiIndex.from_product([["Direction"], [_ for _ in range(num_directions)] + ['Mean Direction', 'Direction Standard Deviation']])
    div_iter = pd.MultiIndex.from_product([["Divergence"], [_ for _ in range(num_divergences)] + ['Mean Divergence', 'Divergence Standard Deviation']])

    speed_df = pd.DataFrame(combined_speeds, index = frame_indices, columns = speed_iter)
    direction_df = pd.DataFrame(combined_directions, index = frame_indices, columns = dir_iter)
    divergence_df = pd.DataFrame(combined_divergences, index = frame_indices, columns = div_iter)

    output_file = file_location if file_location else f'{os.path.dirname(path)}/optical_flow.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name = 'optical_flow_summaries', index = False)
        speed_df.to_excel(writer, sheet_name = 'speed_distribution', index=True)
        direction_df.to_excel(writer, sheet_name = 'direction_distribution', index = True)
        divergence_df.to_excel(writer, sheet_name = 'divergence_distribution', index = True)

    return summary_df, speed_df, direction_df, divergence_df

def binarized(path, file_location = None):
    frames, frame_idx = read_bin_frames(path)
    areas_lst = []
    a_factors = []
    num_islands = []
    directions = []
    island_distances = []
    
    total_areas = []
    total_anistropies = []
    total_directions = []
    total_distances = []

    for i in range(len(frames)):
        areas = get_island_area_stats(frames[i])
        i_distances = list(get_island_distances(frames[i]).values())
        dirs, af = get_island_orientation(frames[i])
        
        num_islands.append(len(areas))
        
        areas_lst.append(np.unique(np.round(areas, 2), return_counts = True))
        a_factors.append(np.unique(np.round(af, 2), return_counts = True))
        directions.append(np.unique(np.round(dirs, 1), return_counts = True))
        i_dist = np.histogram(i_distances, bins = 100)
        island_distances.append((i_dist[1][:-1], i_dist[0][:-1]))

        total_areas.extend(np.round(areas, 2))
        total_anistropies.extend(np.round(af, 2))
        total_directions.extend(np.round(dirs, 1))
        total_distances.extend(i_distances)

    areas_lst.append(np.unique(np.round(total_areas, 2), return_counts = True))
    a_factors.append(np.unique(np.round(total_anistropies, 2), return_counts = True))
    i_dist = np.histogram(total_distances, bins = 100)
    island_distances.append((i_dist[1][:-1], i_dist[0][:-1]))

    frame_indices = pd.MultiIndex.from_product([sorted(frame_idx) + ['Total'], ['', 'Count']])
    # print(frame_indices)
    num_areas = len(areas_lst[-1][0])
    num_af = len(a_factors[-1][0])
    num_id = 100
    areas_lst, areas_count = pad_for_df(areas_lst)
    a_factors, af_count = pad_for_df(a_factors)
    island_distances, distance_count = pad_for_df(island_distances)

    combined_areas = []
    combined_af = []
    combined_id = []

    for a, ac, af, afc, id, idc in zip(areas_lst, areas_count, a_factors, af_count, 
    island_distances, distance_count):
        combined_areas.extend([a, ac])
        combined_af.extend([af, afc])
        combined_id.extend([id, idc])

    summary_df = pd.DataFrame({
        'Frames' : sorted(frame_idx) + ['Total'],
        'Number of Islands' : num_islands + [''],
        'Mean Island Area' : [np.nanmean(a) for a in areas_lst],
        'Maximum Island Area' : [max(a) for a in areas_lst],
        'Mean Island Separation' : [np.nanmean(i_dist) for i_dist in island_distances],
        'Mean Anisotropy Factor' : [np.nanmean(af) for af in a_factors]
    })

    area_iter = pd.MultiIndex.from_product([['Island Area'], [_ for _ in range(num_areas)]])
    af_iter = pd.MultiIndex.from_product([['Anisotropy Factor'], [_ for _ in range(num_af)]])
    id_iter = pd.MultiIndex.from_product([['Island Distances'], [_ for _ in range(num_id)]])

    areas_df = pd.DataFrame(combined_areas, index = frame_indices, columns = area_iter)
    af_df = pd.DataFrame(combined_af, index = frame_indices, columns = af_iter)
    id_df = pd.DataFrame(combined_id, index = frame_indices, columns = id_iter)

    output_file = file_location if file_location else f'{os.path.dirname(path)}/binarization.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name = 'binarization_summaries', index = False)
        areas_df.to_excel(writer, sheet_name = 'island area distribution', index = True)
        af_df.to_excel(writer, sheet_name = 'anisotropy_factors', index = True)
        id_df.to_excel(writer, sheet_name = 'island_distances', index = True)

    return summary_df, areas_df, af_df, id_df

def intensity(path, file_location = None):
    frame_dists, frame_nums = read_intensity_dist(path)
    frame_dists_2 = [[[int(i_dist[0])] * int(i_dist[1]) for i_dist in frame] for frame in frame_dists]
    def flatten(xss):
        return np.array([x for xs in xss for x in xs])
    
    frame_dists_flattened = [flatten(frame) for frame in frame_dists_2]
    kurts = [kurtosis(frame) for frame in frame_dists_flattened]
    modeskews = [mode_skew(frame) for frame in frame_dists_flattened]
    medskews = [med_skew(frame) for frame in frame_dists_flattened]

    frame_nums, kurts, modeskews, medskews = zip(*sorted(zip(frame_nums, kurts, modeskews, medskews), key = lambda x: x[0]))

    df = pd.DataFrame({'Frames': frame_nums,
                       'Mode Skewness': modeskews,
                       'Median Skewness': medskews,
                       'Kurtosis': kurts})
    
    output_file = file_location if file_location else f'{os.path.dirname(path)}/intensity_distribution.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name = 'intensity_distribution', index = False)

    return df

def compare_intensity_rds(paths, file_location = None):
    dfs = []
    for path in paths:    
        dfs.append(intensity(path))

    df = pd.concat(dfs, axis = 1, join = 'outer', keys = [f'File {i}: {paths[i - 1]}' for i in range(1, len(paths) + 1)])
    
    output_file = file_location if file_location else f'{os.path.dirname(paths[0])}/intensity_distribution_comparison.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name = 'intensity_distribution', index = True)

def compare_ff_rds(paths, spatial_intervals, time_intervals, size_units, time_units, file_location = None):

    if len(paths) != len(spatial_intervals) or len(paths) != len(time_intervals):
        print('')
        return
    
    dfs = []
    speed_dfs = []
    dir_dfs = []
    div_dfs = []

    for path, space_int, time_int in zip(paths, spatial_intervals, time_intervals):
        df, speed_df, dir_df, div_df = flow(path, space_int, size_units, time_int, time_units)
        speed_df = speed_df.reset_index().rename(columns = {'level_0':'Frames', 'level_1':''})
        dir_df = dir_df.reset_index().rename(columns = {'level_0':'Frames', 'level_1':''})
        div_df = div_df.reset_index().rename(columns = {'level_0':'Frames', 'level_1':''})
        dfs.append(df)
        speed_dfs.append(speed_df)
        dir_dfs.append(dir_df)
        div_dfs.append(div_df)
    
    key_list = [f'File {i}: {paths[i - 1]}' for i in range(1, len(paths) + 1)]

    combined_summary, summary_names = combine_and_export_dfs(dfs, 'optical_flow_data', key_list)
    combined_speeds, speeds_names = combine_and_export_dfs(speed_dfs, 'speed_distribution', key_list)
    combined_dirs, dir_names = combine_and_export_dfs(dir_dfs, 'direction_distribution', key_list)
    combined_divs, div_names = combine_and_export_dfs(div_dfs, 'divergence_distribution', key_list)

    df_lists = [combined_summary, combined_speeds, combined_dirs, combined_divs]
    name_lists = [summary_names, speeds_names, dir_names, div_names]

    output_file = file_location if file_location else f'{os.path.dirname(paths[0])}/optical_flow_comparison.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for df_list, name_list in zip(df_lists, name_lists):
            for df, sheet_name in zip(df_list, name_list):
                # print(sheet_name)
                df.to_excel(writer, sheet_name = sheet_name, index = True)

def compare_binarized_rds(paths, file_location = None):
    dfs = []
    area_dfs = []
    af_dfs = []
    id_dfs = []

    for path in paths:
        df, area_df, af_df, id_df = binarized(path)
        area_df = area_df.reset_index().rename(columns = {'level_0':'Frames', 'level_1':''})
        af_df = af_df.reset_index().rename(columns = {'level_0':'Frames', 'level_1':''})
        id_df = id_df.reset_index().rename(columns = {'level_0':'Frames', 'level_1':''})
        dfs.append(df)
        area_dfs.append(area_df)
        af_dfs.append(af_df)
        id_dfs.append(id_df)
    
    key_list = [f'File {i}: {paths[i-1]}' for i in range(1, len(paths) + 1)]

    combined_summary, summary_names = combine_and_export_dfs(dfs, 'binarization_data', key_list)
    combined_areas, area_names = combine_and_export_dfs(area_dfs, 'island_areas', key_list)
    combined_afs, af_names = combine_and_export_dfs(af_dfs, 'anisotropy_factors', key_list)
    combined_ids, id_names = combine_and_export_dfs(id_dfs, 'island_distances', key_list)

    df_lists = [combined_summary, combined_areas, combined_afs, combined_ids]
    name_lists = [summary_names, area_names, af_names, id_names]

    output_file = file_location if file_location else f'{os.path.dirname(paths[0])}/binarization_comparisons.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for df_list, name_list in zip(df_lists, name_lists):
            for df, sheet_name in zip(df_list, name_list):
                # print(sheet_name)
                df.to_excel(writer, sheet_name = sheet_name, index = True)

def combine_and_export_dfs(df_list, df_name, key_list):
    MAX_EXPORT_COLUMNS = 16384 # Maximum number of columns Excel allows in a single sheet
    df_len = [len(df.columns) for df in df_list]
    indices = []
    df_len_combined = np.cumsum(df_len)

    # While any values of df_len_combined > 0:
    #   find the first value that is >= MAX_EXPORT_COLUMNS -> IDX
    #   add IDX to indices array
    #   subtract all values in df_len_combined by value

    while (df_len_combined > 0).any():
        # print(df_len_combined)
        if (df_len_combined < MAX_EXPORT_COLUMNS).all():
            # print('Early Break: No partitioning needed')
            break
        index = np.argmax(df_len_combined >= MAX_EXPORT_COLUMNS)
        # if (df_len_combined >= MAX_EXPORT_COLUMNS).all() == False:
        #     continue
        if (not (index - 1) in indices) and index != 0:
            indices.append(index - 1)
        value = df_len_combined[index]
        indices.append(index)
        df_len_combined = df_len_combined - value * np.ones(df_len_combined.shape)

    if 0 in indices:
        indices.remove(0)
    indices.append(len(df_list))

    # print(indices)

    if len(indices) == 1:
        combined_df = pd.concat(df_list, axis = 1, join = 'outer', keys = key_list, sort = True)
        return [combined_df], [df_name]
    
    idx1 = 0
    combined_dfs = []
    df_names = []
    for i in range(len(indices)):
        idx2 = indices[i]
        idx_range = df_list[idx1:idx2]
        key_range = key_list[idx1:idx2]
        if idx1 != idx2:
            combined_df = pd.concat(idx_range, axis = 1, join = 'outer', keys = key_range, sort = True)
        else:
            combined_df = df_list[idx1]
        file_number = f'File {idx1 + 1}-{idx2 + 1}' if idx2 - idx1 > 1 else f'File {idx2}'
        df_names.append(f'{df_name} - {file_number}')
        combined_dfs.append(combined_df)
        idx1 = idx2

    return combined_dfs, df_names
