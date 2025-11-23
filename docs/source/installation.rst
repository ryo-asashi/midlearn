Installation
============

You can install the development version of **midlearn** from GitHub with:

   .. code:: console

      pip install git+https://github.com/ryo-asashi/midlearn.git

or from PyPI:

   .. code:: console

      pip install midlearn

Requirements
------------

This Python library is a **rpy2**-based wrapper and requires a working R installation on your system, as well as the **midr** package.

You can install the R package from CRAN by running the following command in your R console:

   .. code:: r

      install.packages('midr')

   .. warning::

      **R Installation:** Before installing the Python package, ensure that R is properly installed on your system. 
      On some operating systems (particularly Windows), you may need to set the ``R_HOME`` environment variable manually for **rpy2** to find your R installation.
