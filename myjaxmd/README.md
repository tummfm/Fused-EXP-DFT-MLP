# Differentiable Trajectory Reweighting

Code of the paper "Learning neural network potentials from 
experimental data via Differentiable Trajectory Reweighting".

The Jupyter notebooks provide a guided tour through the code and 
the test cases considered in this study. Running the toy example 
will take around 2 hours, training the diamond and the water example 
takes several days. However, training can be skipped by loading our 
obtained results.

## Installation

1. Create virtual environment with Python 3.8 and activate it
```
virtualenv -p /usr/bin/python3.8 venv
```
2. All dependencies will be installed automatically with the following command: <br/>
```
pip install --upgrade pip
pip install -e .[all]
```
3. However, this only installs a CPU version of Jax. To enable GPU support, 
please override the jaxlib version, e.g. with the following command:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
## Tutorial

The provided Jupyter notebooks provide a guided tour through the 
DiffTRe method and its application.
To run the notebooks, please install the Jupyter package.

## Build documentation

Browsing through the auto-generated documentation files if often more convenient
than looking for the documentation in the code. You can build the documentation 
by installing the packages required by sphinx
```
cd docs/
pip install -r requirements.txt
```
and building the documentation in the target format, e.g.
```
make html
```
You can view the documentation by opening a .html file from 
```docs/build/html``` in your browser.


## Python programming style
This repository tries to follow the 
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).
Please take advantage of the automatic code inspection via pylint.
For PyCharm, you can use the plugin as described 
[here](https://github.com/leinardi/pylint-pycharm).


## Git configuration
Please change the git config in your local repository such that each student 
can push with its own credentials:
```
$ git config user.name "John Doe"
$ git config user.email johndoe@example.com
```
## Remote ssh via PyCharm
These instructions work for PyCharm, bit VSCode would work similarly.

1. On your computer: Create a directory where you want to the install 
the project and clone the project with git clone git@gitlab.lrz.de:ga38jij/myjaxmd.git.
2. On server: Create a directory where you want to have code and data and create 
the virtual environment in there (Installation 1.).
3. On your computer: With pycharm create ssh connection and use python3 from 
venv, also set correct path. Following this link
https://medium.com/@stano/editing-and-executing-remote-python-code-in-pycharm-225e63a519b4 
we do Preferences -> Project -> Project Interpreter -> SSH Interpreter 
-> New server configuration -> next -> Interpreter choose python3 of your 
virtual environment -> Sync folder: choose desired folder on server -> Finish 
-> Apply -> Done.
4. On server: Finish pip installation (Installation 2. and 3.) 

## Overwrite jax-sgmc with full package
If ỳou need the full jax-sgmc package, e.g. if you're training via SG-MCMC, you
can install if from your local repository (as it is not yet hosted on PyPi): e.g.
```
pip install git+file:///home/stephan/PycharmProjects/jax-sgmc
```
Alternatively, you can also install it directly from Github:
```
pip install git+https://github.com/tummfm/jax-sgmc.git
```
## Debugging

Memory profiling:
    https://jax.readthedocs.io/en/latest/device_memory_profiling.html
    install as admin:
        install Graphviz: sudo apt-get install graphviz
        install Go: sudo snap install --classic go (see, in case: https://github.com/golang/go/wiki/Ubuntu)
    as user:
        install pprof: go get -u github.com/google/pprof
        run pprof:  ~/go/bin/pprof --web memory.prof

profile runtime:
python -m cProfile -s cumtime myscript.py