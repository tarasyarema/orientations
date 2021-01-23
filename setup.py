# -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup
from codecs import open
from setuptools.command.test import test as TestCommand


def readfile(filename):
    with open(filename,  encoding='utf-8') as f:
        return f.read()


class SageTest(TestCommand):
    def run_tests(self):
        errno = os.system("sage -t --force-lib orientations")
        if errno != 0:
            sys.exit(1)


setup(
    name="orientations",
    version=readfile("VERSION").strip(),
    description='Enumerating k-connected orientations',
    long_description=readfile("README.md"),
    long_description_content_type="text/markdown",
    url='https://github.com/tarasyarema/orientations',
    author='Taras Yarema',
    author_email='tarasyarema@pm.me',
    license='MIT',
    # classifiers list:
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords="SageMath packaging",
    packages=['orientations'],
    cmdclass={'test': SageTest},
    setup_requires=['sage-package'],
    install_requires=['sage-package', 'sphinx'],
)
