"""Setup for pip installation"""

from setuptools import setup, find_packages

import mlpy


with open('README.md') as file:
    readme = file.read()

requirements = [
    'numpy>=1.24.3',
    'matplotlib>=3.7.2'
]

test_requirements = [
    'pytest'
]

setup(
    name='mlpy',
    version=mlpy.__version__,
    description='A beginner-friendly machine learning library in Python',
    long_description=readme,
    author='Felix Waiblinger',
    author_email='felix.waiblinger@gmail.com',
    url='TODO',
    download_url='https://github.com/FelixWaiblinger/MLpy',
    license='BSD',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>3.11.0',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11'
    ],
    test_suite='tests',
    tests_require=test_requirements
)