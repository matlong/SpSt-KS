"""
Run 2D Kuramoto–Sivashinsky (KS) equation in nonconservative form 
using `KS2D_h` class from KS_solver.
"""

import numpy as np
import torch
from KS_solver import KS2D_h

torch.backends.cudnn.deterministic = True

# Set model param
param = {
    'Lx': 1e3, # x grid length
    'Ly': 1e3, # y grid length
    'nx': int(1e3), # x grid points
    'ny': int(1e3), # y grid points
    'n_ens': 1, # ensemble size
    'dt': 0.5, # timestep
    'dtype': torch.float64, # torch.float32 or torch.float64
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
}
ks2 = KS2D_h(param)

# Set initital condition
t = 0
ks2.h = 2e-2*(torch.rand_like(ks2.h)-0.5) # random pertubation
#ks2.h[:,:,:] = torch.sin(ks2.x) * torch.cos(ks2.y)

# Set control param
n_steps = int(1e4/ks2.dt)+1
freq_checknan = 100
freq_log = 100
freq_plot = 100
diag_spec = True
diag_t0 = 100.
freq_save = n_steps//1000
n_steps_save = 0
outdir = f'data/KS2D_h/Lx{int(ks2.Lx)}_Ly{int(ks2.Ly)}'

# Init. output
if freq_save > 0:
    import os
    os.makedirs(outdir) if not os.path.isdir(outdir) else None
    filename = os.path.join(outdir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(outdir, f'h_{n_steps_save}.npz')
    np.savez(filename, t=float(t), h=ks2.h.cpu().numpy())
    n_steps_save += 1

# Init. figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()  
    fig, ax = plt.subplots(constrained_layout=True, subplot_kw={'projection':'3d'})
    ax_kwargs = dict(
        xlabel=r'$x$',
        ylabel=r'$y$',
        zlabel=r'$h$',
    )
    surf_kwargs = {
            'cmap': 'viridis', 
            'rstride': 1, 
            'cstride': 1,
            }
    x, y = ks2.x.cpu().numpy(), ks2.y.cpu().numpy()
    surf = ax.plot_surface(x, y, (ks2.h[0]).cpu().numpy(), **surf_kwargs)
    ax.set(**ax_kwargs, title=f't = {t:.2f}') 
    plt.pause(0.05)

if diag_spec:
    hpsd = torch.zeros_like(ks2.calc_spec(ks2.h[0])) 
    npsd = 0

# Time-stepping
for n in range(1, n_steps+1):
    ks2.step()
    t += ks2.dt

    if n % freq_checknan == 0 and torch.isnan(ks2.h).any():
        raise ValueError('Stopping, NAN number at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        surf.remove()
        surf = ax.plot_surface(x, y, (ks2.h[0]).cpu().numpy(), **surf_kwargs)
        ax.set_title(f'$t = {t:.2f}$')
        fig.canvas.draw_idle()
        plt.pause(0.01)

    if freq_log > 0 and n % freq_log == 0:
        log_str = f'{n=:06d}, t={t:.3f}, umean={ks2.h.mean().cpu().numpy():+.2E}, hmax={ks2.h.max().cpu().numpy():.2E}, hrms={(ks2.h**2).mean().sqrt().cpu().numpy():.2E}.'
        print(log_str)

    if freq_save > 0 and n % freq_save == 0:
        filename = os.path.join(outdir, f'h_{n_steps_save}.npz')
        np.savez(filename, t=float(t), h=ks2.h.cpu().numpy())
        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'saved h to {filename}')
        
    if diag_spec and t >= diag_t0:
        h = ks2.h - ks2.h.mean(dim=(-2,-1))
        hpsd += ks2.calc_spec(h[0]) 
        npsd += 1

print('********** END OF SIMULATION **********')

if diag_spec:
    k = abs(ks2.kr).cpu().numpy()
    hpsd = (hpsd/npsd).cpu().numpy()
    
    import matplotlib.pyplot as plt
    fig_, ax_ = plt.subplots(tight_layout=True)
    ax_.loglog(k, hpsd)
    ax_.set(xlabel=r'$\kappa$', ylabel=r'PSD ($\kappa$)', title='Mean spectrum')
    ax_.grid(which='both', axis='both')
    plt.show()
