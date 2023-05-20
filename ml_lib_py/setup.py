from setuptools import setup, find_packages

setup(
    name='ml_lib_py',
    version='0.1.0',
    description='some description',
    url='https://github.com/FelixWaiblinger/ml-lib-py',
    author='Felix Waiblinger',
    author_email='felix.waiblinger@gmail.com',
    license='',
    packages=find_packages(),
    install_requires=[
        #'shutils',
        'requests',
        'numpy==1.16',
        'python-mnist'
    ],
    python_requires='<3',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ]
)