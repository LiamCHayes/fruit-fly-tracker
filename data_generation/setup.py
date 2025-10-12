"""Make the utilities pip installable for use in the data generation"""

from setuptools import setup, find_packages

setup(
    name="utilities",
    version="0.1",
    packages=find_packages(),
)

