import os

from setuptools import setup
from setuptools import find_packages

readme = open("README.md").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

required_packages = open('requirements.txt').read().splitlines()

setup(
	name='ddprism',
	version='1.0.0',
	description='Joint posterior sampling with diffusion priors',
	long_description=readme,
	author='ddprism developers',
	url='https://github.com/swagnercarena/ddprism',
	packages=find_packages(PACKAGE_PATH),
	package_dir={'ddprism': 'ddprism'},
	include_package_data=True,
	install_requires=required_packages,
	license='Apache2.0',
	zip_safe=False
)
