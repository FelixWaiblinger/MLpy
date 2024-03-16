from setuptools import setup, find_packages

setup(
    name='mlpy',
    version='0.1.0',
    description='some description',
    url='https://github.com/FelixWaiblinger/MLpy',
    author='Felix Waiblinger',
    author_email='felix.waiblinger@gmail.com',
    license='',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    python_requires='<3',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12'
    ]
)