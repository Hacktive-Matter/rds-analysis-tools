import numpy as np
import matplotlib.pyplot as plt
from import_rds import read_flow_fields, read_bin_frames
import os

def divergence_curl_2d(flow):
    flow = np.swapaxes(flow, 0, 1)
    f = [flow[:,:,0], flow[:,:,1]]
    theta = np.arctan2(flow[:,:,1], flow[:,:,0])
    angles = [np.cos(theta), np.sin(theta)]
    divergence = np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(len(f))])
    radial_divergence = np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) * angles[i] for i in range(len(f))])
    curl = np.ufunc.reduce(np.subtract, [np.gradient(f[len(f)-1-i], axis=i) for i in range(len(f))])
    return divergence, curl, radial_divergence

def get_divergence(path, path2):
    flow_fields, indices = read_flow_fields(path)
    bin_frames, indices2 = read_bin_frames(path2)
    print(len(flow_fields), len(bin_frames))
    mid_points = [np.mean(index) for index in indices]
    divergences = []
    for i in range(len(flow_fields)):
        flow_field = flow_fields[i] * np.stack([bin_frames[i + 1]] * 2, axis = 2)
        divergences.append(divergence_curl_2d(flow_field)[0].mean())
    plt.plot(mid_points, divergences, "b-")
    plt.hlines(0, np.min(mid_points), np.max(mid_points))
    plt.xticks(np.arange(0, np.max(mid_points), 5))
    plt.xlabel('Time (Frames)')
    plt.ylabel('Divergence')
    plt.savefig(os.path.join(os.path.dirname(path), 'divergence_vs_time.png'))
    plt.close('all')
    plt.clf()