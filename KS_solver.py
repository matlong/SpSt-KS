"""
PyTorch solver of 1D & 2D Kuramoto–Sivashinsky (KS) equations in periodic domain (0,2π),
using pseudo spectral method and Strang splitting expontential integrator.
A stochastic KS with an additive random forcing on the RHS can also be solved.
"""

import numpy as np
import torch

class KS1D_h:
    """
    Solver of 1D KS equation in nonconservative/potential form.
    - Physical-space form:
      ∂h/∂t = - h_xx - h_xxxx - ½ (h_x)²
    - Spectral-space form:
      ∂ĥ_k/∂t = (k² - k⁴) ĥ_k - ½ 𝔉{(h_x)²}_k
    """

    def __init__(self, param):
        # Input parameters
        self.nx = param['nx']
        self.n_ens = param['n_ens']
        self.dt = param['dt']
        self.Lx = param['Lx']
        self.rand_forcing = param['rand_forcing']
        if self.rand_forcing:
            self.sigma = param['sigma'] # amplitude of random forcing
            self.half_range = self.nx//2-1 if self.nx % 2 == 0 else self.nx//2 # used for Hermitian noise
        self.dtype = param['dtype']
        self.device = param['device']
        self.arr_shape = (self.n_ens, self.nx)
        self.arr_kwargs = {
                'dtype': self.dtype,
                'device': self.device
                }

        # Initializations
        self.init_grid()
        self.init_step()
        self.h = torch.zeros(self.arr_shape, **self.arr_kwargs)

    def init_grid(self):
        # Spatial grid
        self.dx = self.Lx/self.nx
        self.x = torch.linspace(self.dx/2, self.Lx-self.dx/2, self.nx, **self.arr_kwargs)

        # Spectral grid
        self.k = torch.fft.fftfreq(self.nx, self.dx/(2*np.pi), **self.arr_kwargs)

        # Anti-aliasing using 2/3 rule
        self.mask_dealias = (abs(self.k) < (2/3)*abs(self.k).max()) 

    def init_step(self):
        """Initialize time-stepping operators."""
        # RHS linear operator 
        A = self.k**2 - self.k**4 

        # Discret semi-group operator [Brachet et al. (2020)]
        self.exp_int = torch.exp(A * self.dt/2) 
        
        # Std. of stochastic convolution integral
        if self.rand_forcing:
            var = torch.zeros_like(A)
            var[0] = self.dt # limit for k=0
            var[1:] = (torch.exp(2*A[1:] * self.dt) - 1) / (2*A[1:]) # k~=0
            self.std_sto_int = self.sigma * var**(1./2)

    def nonlinear_rhs(self, h_hat):
        """Compute the RHS nonlinear term in spectral space."""
        N = (torch.fft.ifft(1j*self.k*h_hat).real)**2/2
        N_mean = N.mean(dim=-1, keepdim=True) 
        N -= N_mean # remove mean
        N_hat = self.mask_dealias * torch.fft.fft(N) # anti-aliasing
        N_hat[:,0] = 0. # for numerical precision
        return -N_hat, -N_mean

    def sto_conv_int(self):
        """Compute the stochastic convlution integral due to the additive random forcing."""
        # Generate Hermitian Gaussian profile
        z_hat = torch.zeros_like(self.h) + 1j*torch.zeros_like(self.h)
        z_hat[:,0] = torch.randn(self.n_ens, **self.arr_kwargs) # pure real for k=0
        z_hat[:,self.nx//2] = torch.randn(self.n_ens, **self.arr_kwargs) # pure real for Nyquist mode
        r = torch.randn((self.n_ens, self.half_range), **self.arr_kwargs)
        s = torch.randn_like(r) # indep. real & imag parts
        z_half = (r + 1j*s) / 2.**(1/2)
        z_hat[:,1:1+self.half_range] = z_half
        z_hat[:,-self.half_range:] = z_half.flip(dims=[-1]).conj()

        # Check Hermitianity
        #print(abs(torch.fft.ifft(z_hat).imag).max()) # it should be near zero

        # Scale by std. of the stochastic integral
        return self.std_sto_int * z_hat
   
    def step(self):
        """Strang splitting expontential integrator [Brachet-et-al., 2020].
        Notes on mean (k=0) behaviour:
        - Linear operator vanishes at k=0, so the mean h̄ is driven only by the
          spatial average of the nonlinear term:
            dh̄/dt = - ½ ⟨h_x²⟩.
        - Practically split h = h̄(t) + h'(x,t). Advance h' with exp-RK4 and 
          update h̄ by RK4 using the same nonlinear-stage evaluations for consistency.
        """        
        # Seperation of mean-fluctuation
        h_mean = self.h.mean(dim=-1, keepdim=True)
        h_hat = torch.fft.fft(self.h - h_mean)

        # Prediction step
        h_hat = self.exp_int * h_hat

        # RK4 stepping for nonlinear term of fluctuation
        f1_hat, m1 = self.nonlinear_rhs(h_hat)
        f2_hat, m2 = self.nonlinear_rhs(h_hat + self.dt * f1_hat/2)
        f3_hat, m3 = self.nonlinear_rhs(h_hat + self.dt * f2_hat/2)
        f4_hat, m4 = self.nonlinear_rhs(h_hat + self.dt * f3_hat)
        h_hat += self.dt * (f1_hat + 2*f2_hat + 2*f3_hat + f4_hat)/6        
        h_mean += self.dt * (m1 + 2*m2 + 2*m3 + m4)/6 # update mean        
        
        # Correction step
        h_hat = self.exp_int * h_hat
        if self.rand_forcing:
            h_hat += self.sto_conv_int() # add random forcing
        self.h = torch.fft.ifft(h_hat).real + h_mean


class KS1D_u:
    """
    Solver of 1D KS equation in conservative/slope form.
    - Physical-space form:
      ∂u/∂t = - u_xx - u_xxxx - ½ (u²)_x
    - Spectral-space form:
      ∂û_k/∂t = (k² - k⁴) û_k - (ik/2) 𝔉{u²}_k
    """

    def __init__(self, param):
        # Input parameters
        self.nx = param['nx']
        self.n_ens = param['n_ens']
        self.dt = param['dt']
        self.Lx = param['Lx']
        self.rand_forcing = param['rand_forcing']
        if self.rand_forcing:
            self.sigma = param['sigma'] # amplitude of random forcing
            self.half_range = self.nx//2-1 if self.nx % 2 == 0 else self.nx//2 # used for Hermitian noise

        self.dtype = param['dtype']
        self.device = param['device']
        self.arr_shape = (self.n_ens, self.nx)
        self.arr_kwargs = {
                'dtype': self.dtype,
                'device': self.device
                }

        # Initializations
        self.init_grid()
        self.init_step()
        self.u = torch.zeros(self.arr_shape, **self.arr_kwargs)

    def init_grid(self):
        # Spatial grid
        self.dx = self.Lx/self.nx
        self.x = torch.linspace(self.dx/2, self.Lx-self.dx/2, self.nx, **self.arr_kwargs)

        # Spectral grid
        self.k = torch.fft.fftfreq(self.nx, self.dx/(2*np.pi), **self.arr_kwargs)

        # Anti-alasing using 2/3 rule
        mask = (abs(self.k) < (2/3)*abs(self.k).max()) 
        self.ik_dealias = 1j*self.k*mask

    def init_step(self):
        """Initialize time-stepping operators."""
        # RHS linear operator 
        A = self.k**2 - self.k**4 

        # Discret semi-group operator
        self.exp_int = torch.exp(A * self.dt/2) # 1/2 for Strang spliting
   
        # Std. of stochastic convolution integral
        if self.rand_forcing:
            var = torch.zeros_like(A)
            var[0] = self.dt # limit for k=0
            var[1:] = (torch.exp(2*A[1:] * self.dt) - 1) / (2*A[1:]) # k~=0
            self.std_sto_int = self.sigma * var**(1./2)

    def nonlinear_rhs(self, u_hat):
        """Compute RHS nonlinear term in spectral space."""
        return -self.ik_dealias * torch.fft.fft(torch.fft.ifft(u_hat).real**2)/2

    def sto_conv_int(self):
        """Compute the stochastic convlution integral due to the additive random forcing."""
        # Generate Hermitian Gaussian profile
        z_hat = torch.zeros_like(self.u) + 1j*torch.zeros_like(self.u)
        z_hat[:,0] = torch.randn(self.n_ens, **self.arr_kwargs) # pure real for k=0
        z_hat[:,self.nx//2] = torch.randn(self.n_ens, **self.arr_kwargs) # pure real for Nyquist mode
        r = torch.randn((self.n_ens, self.half_range), **self.arr_kwargs)
        s = torch.randn_like(r) # indep. real & imag parts
        z_half = (r + 1j*s) / 2.**(1/2)
        z_hat[:,1:1+self.half_range] = z_half
        z_hat[:,-self.half_range:] = z_half.flip(dims=[-1]).conj()

        # Check Hermitianity
        #print(abs(torch.fft.ifft(z_hat).imag).max()) # it should be near zero

        # Scale by std. of the stochastic integral
        return self.std_sto_int * z_hat

    def step(self):
        """Strang splitting expontential integrator [Brachet-et-al., 2020]"""
        # Prediction by exponential integrator
        u_hat = self.exp_int * torch.fft.fft(self.u)

        # Add nonlinear terms by RK4 scheme
        k1 = self.nonlinear_rhs(u_hat)
        k2 = self.nonlinear_rhs(u_hat + self.dt*k1/2)
        k3 = self.nonlinear_rhs(u_hat + self.dt*k2/2)
        k4 = self.nonlinear_rhs(u_hat + self.dt*k3)
        u_hat += self.dt * (k1 + 2*k2 + 2*k3 + k4)/6        

        # Correction by exponential integrator
        u_hat = self.exp_int * u_hat
        if self.rand_forcing:
            u_hat += self.sto_conv_int() # add random forcing
        self.u = torch.fft.ifft(u_hat).real # back to space
   

class KS2D_h:
    """
    Solver of 2D KS equation in nonconservative/potential form.
    - Physical-space form:
      ∂h/∂t = - Δ h - Δ² h - ½ |∇h|²
    - Spectral-space form:
      ∂ĥ/∂t(k) = (|k|² - |k|⁴) ĥ(k) - ½ 𝔉{|∇h|²}(k)
    """
    
    def __init__(self, param):
        # Input parameters
        self.nx = param['nx']
        self.ny = param['ny']
        self.n_ens = param['n_ens']
        self.Lx = param['Lx']
        self.Ly = param['Ly']
        self.dt = param['dt']
        self.dtype = param['dtype']
        self.device = param['device']
        self.arr_shape = (self.n_ens, self.ny, self.nx)
        self.arr_kwargs = {
                'dtype': self.dtype,
                'device': self.device
                }

        # Initializations
        self.init_grid()
        self.init_step()
        self.init_spec()
        self.h = torch.zeros(self.arr_shape, **self.arr_kwargs)

    def init_grid(self):
        # Spatial grid
        self.dx, self.dy = self.Lx/self.nx, self.Ly/self.ny 
        x = torch.linspace(self.dx/2, self.Lx-self.dx/2, self.nx, **self.arr_kwargs)
        y = torch.linspace(self.dy/2, self.Ly-self.dy/2, self.ny, **self.arr_kwargs)
        self.x, self.y = torch.meshgrid(x, y, indexing='xy')

        # Spectral grid
        kx = torch.fft.fftfreq(self.nx, self.dx/(2*np.pi), **self.arr_kwargs)
        ky = torch.fft.fftfreq(self.ny, self.dy/(2*np.pi), **self.arr_kwargs) 
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')

        # Anti-aliasing using 2/3 rule
        maskx = ( abs(self.kx) < (2/3)*abs(self.kx).max() )
        masky = ( abs(self.ky) < (2/3)*abs(self.ky).max() )
        self.mask_dealias = maskx * masky
        
    def init_step(self):
        # RHS linear operator 
        k2 = self.kx**2 + self.ky**2
        A = k2 - k2**2
        
        # Discret semi-group operator
        self.exp_int = torch.exp(A * self.dt/2) 

    def nonlinear_rhs(self, h_hat):
        """Compute the RHS nonlinear term in spectral space."""
        hx = torch.fft.ifft2(1j*self.kx * h_hat).real 
        hy = torch.fft.ifft2(1j*self.ky * h_hat).real
        N = (hx**2 + hy**2)/2 # gradient squared
        N_mean = N.mean(dim=(-2,-1), keepdim=True) # spatial mean 
        N -= N_mean # fluctuation
        N_hat = self.mask_dealias * torch.fft.fft2(N) # anti-aliasing
        N_hat[:,0,0] = 0. # for numerical precision
        return -N_hat, -N_mean

    def step(self):
        """Strang splitting expontential integrator [Brachet-et-al., 2020]."""        
        # Seperation of mean-fluctuation
        h_mean = self.h.mean(dim=(-2,-1), keepdim=True)
        h_hat = torch.fft.fft2(self.h - h_mean)

        # Prediction step
        h_hat = self.exp_int * h_hat

        # RK4 stepping for nonlinear term of fluctuation
        f1_hat, m1 = self.nonlinear_rhs(h_hat)
        f2_hat, m2 = self.nonlinear_rhs(h_hat + self.dt * f1_hat/2)
        f3_hat, m3 = self.nonlinear_rhs(h_hat + self.dt * f2_hat/2)
        f4_hat, m4 = self.nonlinear_rhs(h_hat + self.dt * f3_hat)
        h_hat += self.dt * (f1_hat + 2*f2_hat + 2*f3_hat + f4_hat)/6        
        h_mean += self.dt * (m1 + 2*m2 + 2*m3 + m4)/6 # update mean        
        
        # Correction step
        self.h = torch.fft.ifft2(self.exp_int * h_hat).real
        self.h += h_mean # add mean

    def init_spec(self):
        # Compute horizontal wavenumbers
        k = (self.kx**2 + self.ky**2).sqrt()

        # Compute isotropic wavenumbers
        dkr = 2**(1/2) # radical spacing
        kmax = min(np.pi/self.dx, np.pi/self.dy) # cutoff
        kr = torch.arange(0, kmax, dkr, **self.arr_kwargs) # left border of bins

        # Compute mask for integration over rings
        kr_bin = torch.cat((kr[:]+dkr, kr[-1:]+2*dkr)) # create binning for spectrums
        self.ind_kr = torch.bucketize(k, kr_bin, right=True).flatten()
        self.filt_bin = (self.ind_kr < kr.shape[0]) & (self.ind_kr >= 0) # keep only numerical freq.
        self.nxny2 = (self.nx*self.ny)**2 # for Parserval due to DFT
        self.kr = kr + dkr/2 # convert left border of the bin to center

    def calc_spec(self, val):
        """Compute power spectrum of vals. 
           Requires to already have bins and corresponding indices inside the bins of k.
           filt_bin corresponds to a mask to exclude values outside of bins."""
        # Compute 2D spectrum
        spec2d = abs(torch.fft.fft2(val))**2 / self.nxny2
        # Compute 1D spectrum
        spec1d = torch.zeros(val[0].shape[:-2] + self.kr.shape, **self.arr_kwargs)
        spec1d.index_add_(-1, self.ind_kr[...,self.filt_bin], 
                          spec2d.flatten(-2,-1)[...,self.filt_bin])
        return spec1d

