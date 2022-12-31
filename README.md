# NeuralWorks

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## install pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com) to ensure that your code is formatted correctly and that the
minimum quality standards are met.

So, before committing any changes:

* Install the `pre-commit` library in your environment (if not already installed)

  ```bash
  # install pre-commit (required)
  # NOTE: it should have already been installed by the requirements-dev.txt file
  pip install --upgrade --no-cache-dir pre-commit
  ```

* Run the following commands at the root of the repository, where the `.pre-commit-config.yaml` file is located

  * install the `pre-commit` hooks

    ```bash
    # install the pre-commit hooks (required)
    pre-commit install
    ```

  * install the environments for the `pre-commit` hooks

    ```bash
    # install environments for all available hooks now (optional)
    # otherwise they will be automatically installed when they are first executed
    pre-commit install-hooks
    ```

  * run the `pre-commit` hooks on all files

    ```bash
    # run the hooks on all files with the current configuration (optional)
    # this is what will happen automatically
    pre-commit run --all-files
    ```

The following are optional commands that can be run at any time if needed:

* Auto-update the `pre-commit` hooks (optional)

  ```bash
  # auto-update the version of the hooks (optional)
  pre-commit autoupdate
  ```

* Installs hook environments overriding existing environments (optional)

  ```bash
  # idempotently replaces existing git hook scripts with pre-commit, and also installs hook environments (optional)
  pre-commit install --install-hooks --overwrite
  ```

* Run individual hooks (optional)

  ```bash
  # run individual hooks with the current configuration (optional)
  pre-commit run <hook_id>
  ```

* Store "frozen" hashes of hook repositories in the configuration file (optional)

  ```bash
  # store "frozen" hashes in rev instead of tag names (optional)
  pre-commit --freeze
  ```

  ```bash
  # alternatively, use autoupdate to update the revs in the config file to the latest versions of the hooks
  # and store "frozen" hashes in rev instead of tag names
  pre-commit autoupdate --freeze
  ```

* Hard clean-up of the local repository (optional)

  ```bash
  # Hard cleanup of the local repository (optional)
  pre-commit uninstall
  pre-commit clean
  pre-commit gc
  pre-commit autoupdate
  pre-commit install
  pre-commit install-hooks --overwrite
  pre-commit run --all-files
  pre-commit gc
  ```

## Installing on MacOS (Apple Silicon)

* Install `Xcode Command Line Tools` (it can also be installed from the App Store) - it may take a while

  ```bash
  xcode-select --install
  ```

* (SKIP) Install [`Miniforge`](<https://github.com/conda-forge/miniforge>)

  `Miniforge` is a minimal installer for conda.

  ```bash
  # download the installer
  curl -fsSLo Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"

  # update the permissions of the installer
  chmod -x Miniforge3.sh

  # install it
  # bash Miniforge3.sh

  # non-interactive installation
  #     -b            :   run in batch mode (non-interactive)
  #     -p <prefix>   :   install prefix.
  #     -f            :   force install (overwrite existing files)
  #     -u            :   update existing installation
  #     -s            :   skip PATH modifications
  #     -q            :   quiet mode (no output)
  #     -v            :   verbose mode (maximum output)
  #     -h            :   show this help message and exit
  # Usage:
  #         bash Miniforge3.sh -b -p "${HOME}/conda" -f -u -s -q -v -h
  bash Miniforge3.sh -b -p "${HOME}/conda"

  # create the path to conda and activate conda.
  source "${HOME}/conda/etc/profile.d/conda.sh"

  # activate the base environment
  conda activate
  ```

  Alternatively, you can install `Miniforge` using `Homebrew`:

  ```bash
  brew install miniforge
  ```

* Install [`Mambaforge`](<https://github.com/conda-forge/miniforge>)

  [`Mamba`](https://mamba.readthedocs.io/) is a drop-in replacement for the `conda` package manager that is much faster and uses less memory.

  ```bash
  # download the installer
  curl -fsSLo Mambaforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-$(uname -m).sh"

  # update the permissions of the installer
  chmod -x Mambaforge.sh

  # install it
  # bash Mambaforge.sh

  # non-interactive installation
  #     -b            :   run in batch mode (non-interactive)
  #     -p <prefix>   :   install prefix.
  #     -f            :   force install (overwrite existing files)
  #     -u            :   update existing installation
  #     -s            :   skip PATH modifications
  #     -q            :   quiet mode (no output)
  #     -v            :   verbose mode (maximum output)
  #     -h            :   show this help message and exit
  # Usage:
  #         bash Mambaforge.sh -b -p "${HOME}/conda" -f -u -s -q -v -h
  bash Mambaforge.sh -b -p "${HOME}/conda"

  # create the path to conda and activate conda.
  source "${HOME}/conda/etc/profile.d/conda.sh"

  # Only for the Mambaforge installer
  source "${HOME}/conda/etc/profile.d/mamba.sh"

  # activate the base environment
  conda activate
  ```

* Install [`Homebrew`](<https://brew.sh>)

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

* Install [CMake](<https://cmake.org>) using Homebrew

  ```bash
  brew install cmake
  ```

* Install [OpenMP](<https://www.openmp.org>) using Homebrew

  ```bash
  brew install libomp
  ```

* Install [OpenJDK](<https://openjdk.java.net>) using Homebrew

  * Installation of OpenJDK 11

    ```bash
    # https://github.com/salesforce/Merlion
    brew tap adoptopenjdk/openjdk && brew install --cask adoptopenjdk11
    ```

  * Configuration of OpenJDK 11

    * See <https://stackoverflow.com/a/11372189>

    * See <https://apple.stackexchange.com/a/388623>

    Set the `JAVA_HOME` environment variable to the path of the java executable

    ```bash
    # echo "export JAVA_HOME=\"$(/usr/libexec/java_home)\""
    echo "export JAVA_HOME=\"$(/usr/libexec/java_home)\"" >> ~/.zshenv
    . ~/.zshenv
    ```

    Add java to the `PATH` environment variable

    ```bash
    # echo "export PATH=\"$(/usr/libexec/java_home)/bin:${PATH}\""
    echo "export PATH=\"$(/usr/libexec/java_home)/bin:${PATH}\"" >> ~/.zshenv
    . ~/.zshenv
    ```

    Maybe you need to restart the terminal.

    Finally, check the `JAVA_HOME` and `PATH` environment variables

    ```bash
    echo "${JAVA_HOME}"
    echo "${PATH}"
    ```

<!--
* Install ZMQ using Homebrew (it may be required by jupyter vscode extension for python)

  ```bash
  brew install zeromq
  ```
-->

<!--
  ```bash
  mamba install --quiet --file conda-requirements.txt
  ``` -->

* Create a new `conda` environment (NOTE: `conda-forge` is the default channel)

  ```bash
  # create a new environment from environment.yml

  # source the conda profile script (if not already done)
  source "${HOME}/conda/etc/profile.d/conda.sh"

  # source the mamba profile script (if not already done)
  source "${HOME}/conda/etc/profile.d/mamba.sh"

  # activate the base environment
  conda activate

  # check the current environment (should be base)
  conda env list

  # To find out which python/python3 is being used, use the following commands
  whereis python
  # whereis python3

  # To find out all the python/python3 executables, use the following commands
  whereis -a python
  # whereis -a python3

  # NOTE: pip or pip3 may not be installed in the base environment at this point (so, be careful when using pip or pip3)

  # To find out which pip/pip3 is being used, use the following commands
  whereis pip  # for the base environment, should be: ${HOME}/conda/bin/pip
  # whereis pip3 # for the base environment, should be: ${HOME}/conda/bin/pip3

  # To find out all the pip/pip3 executables, use the following commands
  whereis -a pip
  whereis -a pip3

  # to remove unused packages and caches use the following command
  # (NOTE: '--all' Removes index cache, lock files, unused cache packages, tarballs, and logfiles)
  # conda clean --yes --all

  # to restore the base environment to its original state (if needed) use the following commands
  # first, list the revisions (if any) (NOTE: the last revision is the current state of the original base environment)
  # conda list --revisions
  # then, restore the last revision (if any) (e.g. revision 9)
  # conda install --revision 9

  # update the conda base environment
  mamba update --yes --name base --all

  # update pip, setuptools, and wheel in the base environment
  mamba update --yes --name base pip setuptools wheel

  # update conda in the base environment
  mamba update --yes --name base -c conda-forge conda

  # list the installed packages in the base environment
  mamba list --name base

  # Change the current working directory to the project directory
  cd "${HOME}/Documents/GitHub/neuralworks"

  # remove the old environment (if it exists)
  # conda env remove --name neuralworks

  # create a new environment from environment.yml
  mamba create --yes --name neuralworks python=3.8

  # activate the new environment
  mamba activate neuralworks

  # check the current environment (should be neuralworks)
  mamba env list

  # check the current python being used
  whereis python  # for the 'neuralworks' environment, should be "${HOME}/conda/envs/neuralworks/bin/python"

  # check the current pip being used
  whereis pip  # for the 'neuralworks' environment, should be "${HOME}/conda/envs/neuralworks/bin/pip"
  # python -m pip --version

  # list the installed packages in the new environment (created)
  mamba list --name neuralworks

  # install PyCaret in the new environment (without dependencies)
  # pip install --no-dependencies --ignore-installed 'pycaret==3.0.0rc4'
  # pip install --no-dependencies --ignore-installed 'pycaret==3.0.0rc5'
  # pip install --no-dependencies --ignore-installed 'pycaret==3.0.0rc6'
  python -m pip install --no-dependencies --ignore-installed 'pycaret==3.0.0rc6'

  # deactivate the current environment
  conda deactivate

  # update the new environment from environment.yml
  mamba env update --file environment.yml --prune

  # update the new environment from environment-dev.yml (do not prune, as this will remove packages)
  mamba env update --file environment-dev.yml

  # list the installed packages in the new environment (updated)
  mamba list --name neuralworks

  # init mamba (restart the shell after this)
  mamba init --all

  # Change the current working directory to the project directory
  cd "${HOME}/Documents/GitHub/neuralworks"

  # activate the new environment
  mamba activate neuralworks

  # to update all packages in the environment to their latest compatible versions, use the following command (NOT RECOMMENDED)
  # mamba update --yes --all

  # smoke test the new environment packages
  python -c "import pandas; print('pandas              :   ', pandas.__version__)"
  python -c "import numpy; print('numpy               :   ', numpy.__version__)"
  python -c "import scipy; print('scipy               :   ', scipy.__version__)"
  python -c "import sklearn; print('sklearn             :   ', sklearn.__version__)"
  python -c "import matplotlib; print('matplotlib          :   ', matplotlib.__version__)"
  python -c "import seaborn; print('seaborn             :   ', seaborn.__version__)"
  python -c "import xgboost; print('xgboost             :   ', xgboost.__version__)"
  python -c "import lightgbm; print('lightgbm            :   ', lightgbm.__version__)"
  python -c "import numba; print('numba               :   ', numba.__version__)"
  python -c "import pycaret; print('pycaret             :   ', pycaret.__version__)"
  python -c "import pandas_profiling; print('pandas-profiling    :   ', pandas_profiling.__version__)"
  python -c "import ipywidgets; print('ipywidgets          :   ', ipywidgets.__version__)"

  # check the current pip being used
  whereis pip  # for the 'neuralworks' environment, should be "${HOME}/conda/envs/neuralworks/bin/pip"
  # whereis python
  # python -m pip

  # install the package
  # pip install --editable "${HOME}/Documents/GitHub/neuralworks"
  python -m pip install --editable "${HOME}/Documents/GitHub/neuralworks"

  # enable ipywidgets in the new environment (optional)
  # when using virtual environments, the recommended way to enable ipywidgets
  # is to use the --sys-prefix flag instead of --user
  # https://stackoverflow.com/a/74069936/15999297

  # install the ipywidgets extension
  # jupyter nbextension install widgetsnbextension --py --user  
  jupyter nbextension install widgetsnbextension --py --sys-prefix  # for virtual environments

  # to initialize this nbextension in the browser every time the notebook (or other app) loads,
  # use the following command
  # jupyter nbextension enable widgetsnbextension --py --user
  jupyter nbextension enable widgetsnbextension --py --sys-prefix  # for virtual environments

  # create notebook kernel for the new environment (optional)
  # python -m ipykernel install --user --name yourenvname --display-name 'display-name'
  # python -m ipykernel install --user --name neuralworks --display-name 'neuralworks'
  ```

* Smoke the installation of the new environment packages

  ```bash
  # Change the current working directory to the project directory
  cd "${HOME}/Documents/GitHub/neuralworks"

  # activate the conda environment
  conda activate neuralworks

  # smoke test the new environment packages
  python -c "import pandas; print('pandas              :   ', pandas.__version__)"
  python -c "import numpy; print('numpy               :   ', numpy.__version__)"
  python -c "import scipy; print('scipy               :   ', scipy.__version__)"
  python -c "import sklearn; print('sklearn             :   ', sklearn.__version__)"
  python -c "import matplotlib; print('matplotlib          :   ', matplotlib.__version__)"
  python -c "import seaborn; print('seaborn             :   ', seaborn.__version__)"
  python -c "import xgboost; print('xgboost             :   ', xgboost.__version__)"
  python -c "import lightgbm; print('lightgbm            :   ', lightgbm.__version__)"
  python -c "import numba; print('numba               :   ', numba.__version__)"
  python -c "import pycaret; print('pycaret             :   ', pycaret.__version__)"
  python -c "import pandas_profiling; print('pandas-profiling    :   ', pandas_profiling.__version__)"
  python -c "import ipywidgets; print('ipywidgets          :   ', ipywidgets.__version__)"
  ```

## 3.2. Install the package in `editable` mode

```bash
# Change the current working directory to the project directory
cd "${HOME}/Documents/GitHub/neuralworks"

# activate the conda environment
conda activate neuralworks

# install the package
python -m pip install --editable "${HOME}/Documents/GitHub/neuralworks"
```

## Some libraries used

* <https://pandas.pydata.org/docs/getting_started/install.html>

-------

### python info

```bash
python --version
which python
python -c 'import sys; print(sys.executable)'
```

```bash
python3 --version
which python3
python3 -c 'import sys; print(sys.executable)'
```
