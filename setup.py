from setuptools import setup, Extension
from pathlib import Path


_C_DIRECTORIES = ['src/libift',
                  'src/pyift_c_ext',
                  ]


# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
def _my_build_ext(params):
    from setuptools.command.build_ext import build_ext

    class BuildExt(build_ext):
        def finalize_options(self):
            build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            import builtins
            builtins.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    return BuildExt(params)


def _get_includes():
    includes = _C_DIRECTORIES
    return includes


def _get_sources():
    sources = []
    for directory in _C_DIRECTORIES:
        sources += [str(c_files) for c_files in Path(directory).glob('*.c')]
    return sources


def setup_package():
    build_requires = ['setuptools_scm']
    try:
        import numpy
    except ImportError:
        build_requires.append('numpy')

    exts = [Extension('_pyift',
                      sources=_get_sources(),
                      include_dirs=_get_includes(),
                      extra_compile_args=['-std=gnu11', '-O3'])
            ]

    meta_data = dict(name='pyift',
                     author='Jordao Bragantini',
                     author_email='jordao.bragantini+pyift@gmail.com',
                     description='Python Image Foresting Transform Library',
                     long_description=open('README.md').read(),
                     long_description_content_type="text/markdown",
                     url='https://github.com/pyift/pyift',
                     license='MIT',
                     packages=['pyift'],
                     cmdclass={'build_ext': _my_build_ext},
                     use_scm_version={'write_to': 'src/pyift/_version.py'},
                     setup_requires=build_requires,
                     install_requires=[
                         'numpy>=2',
                         'scipy',
                     ],
                     python_requires=">=3.9",
                     package_dir={'': 'src'},
                     ext_modules=exts,
                     classifiers=[
                         'Development Status :: 4 - Beta',
                         'Intended Audience :: Developers',
                         'Intended Audience :: Education',
                         'Intended Audience :: Science/Research',
                         'License :: OSI Approved :: MIT License',
                         'Programming Language :: C',
                         'Programming Language :: Python :: 3',
                         'Programming Language :: Python :: 3 :: Only',
                         'Programming Language :: Python :: 3.9',
                         'Programming Language :: Python :: 3.10',
                         'Programming Language :: Python :: 3.11',
                         'Programming Language :: Python :: 3.12',
                         'Programming Language :: Python :: 3.13',
                         'Topic :: Scientific/Engineering',
                         'Topic :: Software Development :: Libraries :: Python Modules',
                     ],
                     )

    setup(**meta_data)


if __name__ == '__main__':
    setup_package()
