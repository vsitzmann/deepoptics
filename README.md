# End-to-end Optimization of Optics and Image Processing for Achromatic Extended Depth of Field and Super-resolution Imaging
Project web-site with more infos and videos: [link](https://vsitzmann.github.io/deepoptics/).

Link to our group website with other VR, AR, and computational imaging projects: [link](http://www.computationalimaging.org/)

This repository contains the code to reproduce the results presented in the following paper:

*Sitzmann, Vincent, et al. "End-to-end Optimization of Optics and Image Processing for Achromatic Extended Depth of Field and Super-resolution Imaging" SIGGRAPH 2018.

If you use this code or dataset in you research, please consider citing our paper with the following Bibtex code:

```
@article{sitzmann2018end,
  title={End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging},
  author={Sitzmann, Vincent and Diamond, Steven and Peng, Yifan and Dun, Xiong and Boyd, Stephen and Heidrich, Wolfgang and Heide, Felix and Wetzstein, Gordon},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={114},
  year={2018},
  publisher={ACM}
}
```
A pre-print is available at this [link](https://dl.acm.org/citation.cfm?id=3201333&picked=formats).

## Dataset
We used the high-resolution images from Jegou et al. (2008) for optimizing the phase masks presented in the paper, though any other set of high-resolution images should work. The images are available for download [here](http://lear.inrialpes.fr/~jegou/data.php).

## Usage
This code was written for python 3.6.3 and Tensorflow 1.4.0. 
I recommend to use conda for dependency management. You can create an environment with name "deepoptics" with all dependencies like so:
```
conda env create -f src/environment.yml
```

Each one of the scripts aedof_diffractive.py, focussing_diffractive.py and aedof_zernike.py reproduces one experiment 
from the paper. They take as command-line parameters the log directory, which all log files and checkpoints will be written to,
and the image directory, which should contain high-resolution jpg images (see "Dataset" above). Example call:
```
python focusing_diffractive.py --log_dir /path/to/logdir --img_dir /path/to/image/dir
```
Training progress can be inspected via tensorboard:
```
tensorboard --logdir=/path/to/logdir
```

## Abstract
In typical cameras the optical system is designed first; once it is fixed, the parameters in the image processing algorithm are tuned to get good image reproduction. In contrast to this sequential design approach, we consider joint optimization of an optical system (for example, the physical shape of the lens) together with the parameters of the reconstruction algorithm. We build a fully-differentiable simulation model that maps the true source image to the reconstructed one. The model includes diffractive light propagation, depth and wavelength-dependent effects, noise and nonlinearities, and the image post-processing. We jointly optimize the optical parameters and the image processing algorithm parameters so as to minimize the deviation between the true and reconstructed image, over a large set of images. We implement our joint optimization method using autodifferentiation to efficiently compute parameter gradients in a stochastic optimization algorithm. We demonstrate the efficacy of this approach by applying it to achromatic extended depth of field and snapshot super-resolution imaging.

