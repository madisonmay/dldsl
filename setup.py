#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup


setup(
    name='dldsl',
    version='0.0.1',
    license='MPLv2',
    description='A tiny einsum-inspired domain specific language for deep learning',
    author='Madison May',
    author_email='madison@pragmatic.ml',
    url='https://github.com/madisonmay/dldsl',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        "lark-parser==0.8.5"
    ],
    tests_require=[
        "pytest==5.4.1"
    ]
)
