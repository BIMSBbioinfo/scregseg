========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/scseg/badge/?style=flat
    :target: https://readthedocs.org/projects/scseg
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/wkopp/scseg.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wkopp/scseg

.. |codecov| image:: https://codecov.io/github/wkopp/scseg/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wkopp/scseg

.. |version| image:: https://img.shields.io/pypi/v/scseg.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/scseg

.. |commits-since| image:: https://img.shields.io/github/commits-since/wkopp/scseg/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/wkopp/scseg/compare/v0.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/scseg.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/scseg

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/scseg.svg
    :alt: Supported versions
    :target: https://pypi.org/project/scseg

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/scseg.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/scseg


.. end-badges

Single cell chromatin segmentation

* Free software: MIT license

Installation
============

::

    pip install scseg

Documentation
=============


https://scseg.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
