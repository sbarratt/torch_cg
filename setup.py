import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

setuptools.setup(
    name="cg_cpp",
    ext_modules=[
        CppExtension('cg_cpp', ['cpp/cg.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setuptools.setup(
    name="torch_cg",
    version="0.0.1",
    author="Shane Barratt",
    author_email="stbarratt@gmail.com",
    description="pytorch conjugate gradient",
    url="https://github.com/sbarratt/torch_cg",
    license="MIT",
    platforms="any",
    packages=["torch_cg"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'numpy',
        'scipy'
    ]
)
