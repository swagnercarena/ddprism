import os

from setuptools import setup
from setuptools import find_packages

readme = open("README.md").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

def read_requirements(filename):
	"""Read requirements from file, filtering out empty lines and comments."""
	with open(filename, 'r') as f:
		lines = f.read().splitlines()
	return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

# Core requirements for basic functionality
required_packages = read_requirements('requirements.txt')

# Optional requirements for experiments and extended functionality
experiments_packages = read_requirements('requirements_experiments.txt')

setup(
	name='ddprism',
	version='1.0.0',
	description='Joint posterior sampling with diffusion priors',
	long_description=readme,
	long_description_content_type='text/markdown',
	author='ddprism developers',
	url='https://github.com/swagnercarena/ddprism',
	packages=find_packages(PACKAGE_PATH),
	package_dir={'ddprism': 'ddprism'},
	include_package_data=True,
	install_requires=required_packages,
	extras_require={
		'experiments': experiments_packages,
		'all': experiments_packages,
	},
	license='Apache2.0',
	zip_safe=False,
	python_requires='>=3.8',
)
