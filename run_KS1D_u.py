"""
Run 1D Kuramoto–Sivashinsky (KS) equation in conservative form
using `KS1D_u` class from KS_solver.
"""

import numpy as np
import torch
from KS_solver import KS1D_u

torch.backends.cudnn.deterministic = True

# Set model param
param = {
    'Lx': 1e4, # length param
    'nx': int(1e4), # grid points
    'n_ens': 1, # ensemble size
    'rand_forcing': True, 
    'sigma': 1e-3, # amplitude of random forcing
    'dt': 1., # timestep
    'dtype': torch.float64, # torch.float32 or torch.float64
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
}
ks1 = KS1D_u(param)

# Set initital condition
t = 0
if ks1.rand_forcing:
    ks1.u[:,:] = 0. # quiesient state for SPDE
else:
    ks1.u = 2e-2*(torch.rand_like(ks1.u)-0.5) # random pertubation for PDE

# Set control param
n_steps = int(1e5/ks1.dt)+1
freq_checknan = 100
freq_log = 1000
freq_plot = 0
diag_spec = True
diag_t0 = 100.
freq_save = n_steps//1000
n_steps_save = 0
outdir = f"data/KS1D_u/L{ks1.nx}"

# Init. output
if freq_save > 0:
    import os
    os.makedirs(outdir) if not os.path.isdir(outdir) else None
    filename = os.path.join(outdir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(outdir, f'u_{n_steps_save}.npz')
    np.savez(filename, t=float(t), u=ks1.u.cpu().numpy())
    n_steps_save += 1

# Init. figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
    plt.ion() 
    fig, ax = plt.subplots()
    plt_kwargs = dict(
        ylim=(-5, 5),
        xlabel=r'$x$',
        ylabel=r'$u$',
        title=f'$t = {t:.2f}$',
    )
    line, = ax.plot(ks1.x.cpu().numpy(), (ks1.u[0]).cpu().numpy())
    ax.set(**plt_kwargs)
    ax.grid(True)
    fig.tight_layout()
    plt.pause(0.1)

if diag_spec:
    upsd = torch.zeros_like(ks1.u)
    npsd = 0

# Time-stepping
for n in range(1, n_steps+1):
    ks1.step()
    t += ks1.dt

    if n % freq_checknan == 0 and torch.isnan(ks1.u).any():
        raise ValueError('Stopping, NAN number at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        line.set_ydata((ks1.u[0]).cpu().numpy())
        ax.set_title(f"$t = {t:.2f}$")
        fig.canvas.draw_idle()
        plt.pause(0.02)

    if freq_log > 0 and n % freq_log == 0:
        log_str = f'{n=:06d}, t={t:.3f}, umean={ks1.u.mean().cpu().numpy():+.2E}, umax={ks1.u.max().cpu().numpy():.2E}, urms={(ks1.u**2).mean().sqrt().cpu().numpy():.2E}.'
        print(log_str)

    if freq_save > 0 and n % freq_save == 0:
        filename = os.path.join(outdir, f'u_{n_steps_save}.npz')
        np.savez(filename, t=float(t), u=ks1.u.cpu().numpy())
        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'saved u to {filename}')

    if diag_spec and t >= diag_t0:
        upsd += abs(torch.fft.fft(ks1.u))**2 / ks1.nx**2
        npsd += 1

print('********** END OF SIMULATION **********')

if diag_spec:
    k = ks1.k.cpu().numpy()
    upsd = (upsd/npsd).mean(dim=0).cpu().numpy()
   
    import matplotlib.pyplot as plt
    fig_, ax_ = plt.subplots(tight_layout=True)
    ax_.loglog(k[k>0], upsd[k>0])
    ax_.set(xlabel='$k$', ylabel='PSD ($k$)', title='Mean spectrum')
    ax_.set_ylim(bottom=1e-8)
    ax_.grid(which='both', axis='both')
    plt.show()
