from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch, os

torch_lib_dir = os.path.dirname(torch._C.__file__)

setup(
    name="hpc",
    ext_modules=[
        CUDAExtension(
            name="hpc",  # matches hpc.cpython-*.so
            sources=[
                "scripts/custom/fc_op.cpp",
                "scripts/custom/fc_kernel.cu",
            ],
            library_dirs=[torch_lib_dir],
            libraries=[
                "torch",           # core C++ API + JIT symbols
                "torch_python",    # the Python‐extension glue, where torch::Library ctor lives
            ],
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_dir}"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)