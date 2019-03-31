import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension

setuptools.setup(name='torch_cg',
      ext_modules=[CppExtension('cg', ['cpp/cg.cpp'])],
      cmdclass={'build_ext': BuildExtension})