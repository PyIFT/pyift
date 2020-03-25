import numpy
from setuptools import setup, Extension
from typing import List
from pathlib import Path


_C_DIRECTORIES = ['src/libift',
                  'src/pyift_c_ext',
                  ]


pkg_vars = {}
with open("src/pyift/_version.py") as fp:
    exec(fp.read(), pkg_vars)


def _get_includes() -> List[str]:
    includes = [numpy.get_include()]
    includes += _C_DIRECTORIES
    return includes


def _get_sources() -> List[str]:
    sources = []
    for directory in _C_DIRECTORIES:
        sources += [str(c_files) for c_files in Path(directory).glob('*.c')]
    return sources


exts = [Extension('_pyift',
                  sources=_get_sources(),
                  include_dirs=_get_includes(),
                  extra_compile_args=['-std=gnu11'])
        ]


setup(name='pyift',
      version=pkg_vars['__version__'],
      author='Jordao Bragantini',
      author_email='jordao.bragantini+pyift@gmail.com',
      description='Python Image Foresting Transform Library',
      long_description=open('README.md').read(),
      url='https://github.com/pyift/pyift',
      license='MIT',
      packages=['pyift'],
      install_requires=['numpy',
                        'scipy',
                        ],
      package_dir={'': 'src'},
      ext_modules=exts,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: C',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      )
