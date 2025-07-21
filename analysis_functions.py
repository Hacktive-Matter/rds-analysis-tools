import numpy as np
from scipy.stats import mode
import os, functools, builtins, nd2
import imageio.v3 as iio

def read_file(file_path):
    print = functools.partial(builtins.print, flush=True)
    acceptable_formats = ('.tif', '.nd2', '.tiff')
    
    if (os.path.exists(file_path) and file_path.endswith(acceptable_formats)) == False:
        return None

    if file_path.endswith('.tif'):
        file = iio.imread(file_path)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file
        if file.shape[3] != min(file.shape):
            file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)
    elif file_path.endswith('.nd2'):
        try:
            with nd2.ND2File(file_path) as ndfile:
                if len(ndfile.sizes) >= 5:
                    raise TypeError("Incorrect file dimensions: file must be time series data with 1+ channels (4 dimensions total)")
                if "Z" in ndfile.sizes:
                    raise TypeError('Z-stack identified, skipping to next file...')
                if 'T' not in ndfile.sizes or len(ndfile.shape) <= 2 or ndfile.sizes['T'] <= 5:
                    raise TypeError('Too few frames, unable to capture dynamics, skipping to next file...')
                if ndfile == None:
                    raise TypeError('Unable to read file, skipping to next file...')
                file = ndfile.asarray()
                file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)

        except Exception as e:
            raise TypeError(e)
            
        if isinstance(file, np.ndarray) == False:
            return None
        
    if (file == 0).all():
        print('Empty file: can not process, skipping to next file...')
        return None

    else:
        return file

def med_skew(frame):
    return 3 * (np.mean(frame) - np.median(frame))/np.std(frame)

def mode_skew(frame):
    def calc_mode(frame):
        mode_result = mode(frame.flatten(), keepdims=False)
        mode_intensity = mode_result.mode if isinstance(mode_result.mode, np.ndarray) else np.array([mode_result.mode])
        mode_intensity = mode_intensity[0] if mode_intensity.size > 0 else np.nan
        return mode_intensity
    return (np.mean(frame) - calc_mode(frame))/np.std(frame)

def groupAvg(arr, N, bin_mask=True):
        result = np.cumsum(arr, 0)[N-1::N]/float(N)
        result = np.cumsum(result, 1)[:,N-1::N]/float(N)
        result[1:] = result[1:] - result[:-1]
        result[:,1:] = result[:,1:] - result[:,:-1]
        if bin_mask:
            result = np.where(result > 0, 1, 0)
        return result

def pad_for_df(dataset, include_mean = False):
    max_length = max([len(data[0]) for data in dataset])
    data_values = [list(data[0]) if len(data[0]) == max_length else list(data[0]) + [np.nan] * (max_length - len(data[0])) for data in dataset]
    data_count = [list(data[1]) if len(data[1]) == max_length else list(data[1]) + [np.nan] * (max_length - len(data[1])) for data in dataset]
    if include_mean:
        data_values = [data + [np.nanmean(data), np.nanstd(data)] for data in data_values]
        data_count = [count + [np.nan, np.nan] for count in data_count]
    return data_values, data_count

def get_num_frames(path):
    original_path = path[:-51] + '.tif'
    if not os.path.exists(original_path):
        original_path = path[:-51] + '.nd2'
    file = read_file(original_path)
    return file.shape[0]