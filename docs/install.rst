------------
Installation
------------

Supported Python and NumPy versions are determined according to the `NEP 29 deprecation policy <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.

Where to get it
===============

conda:

.. code:: bash

    conda install -c conda-forge stumpy

pip:

.. code:: bash

    python -m pip install stumpy

pixi:

.. code:: bash

    pixi add stumpy

uv:

.. code:: bash

    uv add stumpy

From source
===========

To install stumpy from source, first clone the source repository:

.. code:: bash

    git clone https://github.com/stumpy-dev/stumpy.git
    cd stumpy

Next, you'll need to install the necessary dependencies:

conda:

.. code:: bash

    conda install -c conda-forge -y --file requirements.txt
    python -m pip install .

pip:

.. code:: bash

    python -m pip install -r requirements.txt
    python -m pip install .

pixi:

.. code:: bash

    pixi install

uv:

.. code:: bash

    uv sync
