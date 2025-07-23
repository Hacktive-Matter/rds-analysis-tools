from numpy.fft import fft2, ifft2, fftshift
import numpy as np
from import_rds import read_flow_fields
import matplotlib.pyplot as plt
import os

def velocity_correlation(path: str, micron_pix_ratio: float = 1):
    flow_fields, indices = read_flow_fields(path)
    labels = ["ro", "go", "bo"]
    frames = [0, int((len(indices))/2) - 1, len(indices) - 1]
    for i in range(3):
        frame = frames[i]
        flow_field = flow_fields[frame]
        vx = flow_field[:,:,0]
        vy = flow_field[:,:,1]
        nx, ny = vx.shape

        disp = np.stack((vx.flatten(), vy.flatten()),axis=1)
        dot_prod = np.sum(disp[None,:,:]*disp[:,None,:],axis=-1)
        norms = np.linalg.norm(disp, axis=1)
        norm_products = norms[None, :] * norms[:, None]
        
        originsx = np.arange(-1*nx/2,nx/2)[:,None].repeat(nx,axis=1)
        originsy = np.arange(-1*ny/2,ny/2)[None,:].repeat(ny,axis=0)
        origins = np.stack((originsx.flatten(), originsy.flatten()),axis=1)
        N = origins.shape[0]
        
        dists = np.linalg.norm(origins[None, :, :] - origins[:, None, :], axis=-1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.where(norm_products > 0, dot_prod / norm_products, 0)
        iu = np.triu_indices(N, k=1)
        dists = dists[iu]
        correlations = correlations[iu]
        
        n_bins = 50
        max_distance = dists.max()
        bins = np.linspace(0, max_distance, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        corr_vs_dist = np.zeros(n_bins)
        for j in range(0, n_bins):
            mask = (dists >= bins[j]) & (dists < bins[j + 1])
            if np.any(mask):
                corr_vs_dist[j] = np.mean(correlations[mask])
            else:
                corr_vs_dist[j] = np.nan
        plt.plot(bin_centers, corr_vs_dist, labels[i], label = f'Frames {indices[frame][0]} to {indices[frame][1]}', alpha = 0.6)
        bottom, _ = plt.ylim()
        plt.ylim(np.min([0, np.nanmin(corr_vs_dist), bottom]), 1.1)

    plt.xlabel('r ($\\mu$m)')
    ylabel = "$\\frac{<V(r) \\cdot V(0)>}{<|V(0)|^2>}$"
    plt.ylabel(ylabel)

    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(path), 'velocity_correlation_curve.png'))
    plt.close('all')
    plt.clf()