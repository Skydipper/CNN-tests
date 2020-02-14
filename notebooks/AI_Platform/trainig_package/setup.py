
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras==2.2.4']

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
