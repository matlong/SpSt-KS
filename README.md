# SpSt-KS

## Description
Project: Spontaneous Stochasticity of the Kuramoto–Sivashinsky (KS) equations. It is for the Fall school in turbulence: October 21-31, 2025, Institut d’Études scientifiques de Cargèse, Corsica, France.  
[https://turbazur.github.io/cargese2025/]

The solver is implemented in PyTorch with pseudo spectral method and Strang splitting expontential integrator.

## Getting Started

### Dependencies

* Prerequisites: PyTorch, Numpy, Matplotlib.

* Tested with Intel CPUs, GPU will also be portable.

### Installing

git clone https://github.com/matlong/SpSt-KS.git

* To install PyTorch with conda for CPU-only users:
```
conda create -n torch python=3.10 -y
conda activate torch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

* To install on GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Executing program

* To run the 1D KS equation in conservative form:
```
python run_ks1d_u.py
```

* To run the 1D KS equation in nonconservative form:
```
python run_ks1d_h.py
```

* The KS equation solvers are in the module `KS_solver.py`.


<!---
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```
-->

## Authors

Long LI

Contact: long.li@inria.fr

<!---
ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)
-->

<!---
## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## Citation

TODO

## Acknowledgments

The authors acknowledge ...
-->

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

<!---
Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
-->
