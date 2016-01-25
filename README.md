[![Build Status](https://travis-ci.org/arvoelke/nengolib.svg?branch=master)](https://travis-ci.org/arvoelke/nengolib) [![codecov.io](https://codecov.io/github/arvoelke/nengolib/coverage.svg?branch=master)](https://codecov.io/github/arvoelke/nengolib?branch=master)

# Nengo Library
Additional extensions for large-scale brain modelling with Nengo.

![img](http://i.imgur.com/moZPzUi.png)

### Highlights
 - `nengolib.Network(...)` serves as a drop-in replacement for `nengo.Network(...)` to improve the performance of an ensemble and the accuracy of its decoders.
 - `nengolib.HeteroSynapse(...)` allows one to connect to an ensemble using a different synapse per dimension or per neuron.
 - `nengolib.LinearFilter(...)` serves as a drop-in replacement for `nengo.LinearFilter(...)` to improve the efficiency of simulations for high-order synapse models.
 - `nengolib.{Lowpass,Alpha,LinearFilter}` are synapses with rich semantics. These linear systems can be scaled, added, multiplied, inverted, compared, and converted between various standard formats. These synapses can also be simulated easily within `Nengo`. For example, to use a double-exponential synapse:

    ```
synapse = (tau1 * nengolib.Lowpass(tau1) - tau2 * nengolib.Lowpass(tau2)) / (tau1 - tau2)
nengo.Connection(a, b, synapse=synapse)
```

### Installation

NengoLib is tested against Python 2.7, 3.4, and 3.5.

To install, first grab the current development version of Nengo [2.1.0-dev](https://github.com/nengo/nengo):
```
git clone https://github.com/nengo/nengo
cd nengo
python setup.py install
```

Then install the development version of NengoLib:
```
git clone https://github.com/arvoelke/nengolib
cd nengolib
pip install -r requirements.txt
python setup.py install
```

We recommend SciPy >= [0.16.0](https://github.com/scipy/scipy/releases/tag/v0.16.0) and NumPy >= [1.9.2](https://github.com/numpy/numpy/releases/tag/v1.9.2), as indicated by `requirements.txt`. If the above `pip install` fails, the Python distribution [Anaconda](https://www.continuum.io/downloads) may be the easiest way to obtain both packages.

On Windows, It may be quicker to pip install the [pre-built Windows binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/) for NumPy and Scipy. For other operating systems, we refer the user to the [SciPy installation guide](http://www.scipy.org/install.html) and the [NumPy installation guide](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html).

### Documentation

Examples and lectures can be found in `doc/notebooks` by running:
```
pip install jupyter
ipython notebook
```
