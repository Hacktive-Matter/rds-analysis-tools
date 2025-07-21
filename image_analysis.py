import csv, os
import numpy as np

'''
Reads flow fields from a BARCODE-generated CSV file into two arrays with dimension (width, height, num_frames)
and one array with the index of the flow fields

@param: path -- the filepath to the CSV file with the flow fields
@return:
- flow_field: the flow field vectors
- flow_field_indices: a list of the first frame f every flow field

'''
def read_flow_fields(path):
    flow_field_x = None
    flow_field_y = None
    flow_field_indices = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        x_flag = True
        x_field = None
        y_field = None
        for row in csvreader:
            if len(row) == 0:
                continue
            elif len(row) == 1:
                if str(row[0]).startswith('X'):
                    x_flag = True
                if str(row[0]).startswith('Y'):
                    x_flag = False
                if str(row[0]).startswith('Flow Field'):
                    flow_idx = int(str(row[0]).split('-')[0].split(' ')[2][1:])
                    flow_idx2 = int(str(row[0]).split('-')[1][:-1])
                    # print(flow_idx, flow_idx2)
                    flow_field_indices.append([flow_idx, flow_idx2])
                    x_flag = True
                    if isinstance(x_field, np.ndarray):
                        flow_field_x = append(flow_field_x, x_field)
                        flow_field_y = append(flow_field_y, y_field)
                        x_field = None
                        y_field = None
            else:
                row = np.array(np.float64(row))
                if x_flag:
                    if not isinstance(x_field, np.ndarray):
                        x_field = row
                    else:
                        x_field = np.vstack((x_field, row))
                        
                else:
                    if not isinstance(y_field, np.ndarray):
                        y_field = row
                    else:
                        y_field = np.vstack((y_field, row))
                        
        if isinstance(x_field, np.ndarray):
            flow_field_x = append(flow_field_x, x_field)
            flow_field_y = append(flow_field_y, y_field)
            x_field = None
            y_field = None
    flow_field_x = np.reshape(flow_field_x, (flow_field_x.shape[0], flow_field_x.shape[1], flow_field_x.shape[2], 1))
    flow_field_y = np.reshape(flow_field_y, (flow_field_y.shape[0], flow_field_y.shape[1], flow_field_y.shape[2], 1))
    flow_field = np.append(flow_field_x, flow_field_y, axis=3)
    return flow_field, flow_field_indices

def read_bin_frames(path):
    bin_frame_indices = []
    bin_frames = None
    with open(path, 'r') as file:
        bin_frame = None
        csvreader = csv.reader(file)
        for row in csvreader:
            if len(row) == 0:
                continue
            if len(row) == 1:
                bin_frame_indices.append(int(row[0]))
                if isinstance(bin_frame, np.ndarray):
                    bin_frames = append(bin_frames, bin_frame)
                    bin_frame=None
            else:
                row = np.array(np.float64(row))
                if not isinstance(bin_frame, np.ndarray):
                    bin_frame = row
                else:
                    bin_frame = np.vstack((bin_frame, row))
        if isinstance(bin_frame, np.ndarray):
            bin_frames = append(bin_frames, bin_frame)
            bin_frame=None
    return bin_frames, bin_frame_indices

def read_intensity_dist(path):
    frame_nums = []
    frame_dists = []
    with open(path, 'r', newline = '\n') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            # print(row)
            if len(row) == 0:
                continue
            if len(row) == 1:
                frame_nums.append(int(row[0].split(' ')[1]))
            else:
                row1 = np.array(row)
                row1 = np.reshape(row1, (row1.shape[0], 1))
                # print(row1)
                # next(csvreader)
                row2 = np.array(next(csvreader))
                row2 = np.reshape(row2, (row2.shape[0], 1))
                # print(row2)
                dist = np.append(row1, row2, axis=1)
                frame_dists.append(dist)
    return frame_dists, np.array(frame_nums)

def append(major_arr, minor_arr):
    if not isinstance(major_arr, np.ndarray):
        major_arr = np.reshape(minor_arr, (1, minor_arr.shape[0], minor_arr.shape[1]))
    else:
        minor_arr = np.reshape(minor_arr, (1, minor_arr.shape[0], minor_arr.shape[1]))
        major_arr = np.append(major_arr, minor_arr, axis=0)
    return major_arr