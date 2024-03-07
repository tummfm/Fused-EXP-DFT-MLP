from setuptools import setup, find_packages

install_requires = [
    'jax>=0.2.12',
    'jaxlib>=0.1.65',
    'jax-md>=0.1.13',
    'optax>=0.0.6',
    'dm-haiku>=0.0.4',
    'sympy>=1.8',
    'tree_math',
    'cloudpickle',
    'chex',
    'blackjax==0.3.0',
    'jaxopt',
    'coax<=0.1.9'  # latest version fails to install
]

extras_requires = {
    'all': ['mdtraj<=1.9.6', 'matplotlib'],
    }

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='chemtrain',
    version='0.1.0',
    license='Apache 2.0',
    description='Train molecular dynamics potentials.',
    author='Stephan Thaler',
    author_email='stephan.thaler@tum.de',
    packages=find_packages(exclude='examples'),
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tummfm/chemtrain',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    zip_safe=False,
)

