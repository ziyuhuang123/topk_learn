import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
VERSION = {}
exec((ROOT / "deep_select" / "__version__.py").read_text(), VERSION)

if CUDA_HOME is None:
    raise RuntimeError("PyTorch must be compiled with CUDA support")

instantiations_dir = ROOT / "csrc" / "cuda_kernels" / "v3" / "instantiations"
instantiation_sources = sorted(instantiations_dir.glob("*.cu"))
if len(instantiation_sources) != 40:
    raise RuntimeError(
        f"Expected 40 CUDA instantiations in {instantiations_dir}, "
        f"found {len(instantiation_sources)}"
    )

sources = [
    "csrc/api.cpp",
    *(str(path.relative_to(ROOT)) for path in instantiation_sources),
]

cuda_archs = [
    arch.strip()
    for arch in os.environ.get("DEEP_SELECT_CUDA_ARCHS", "90a").split(",")
    if arch.strip()
]
if not cuda_archs:
    raise ValueError("DEEP_SELECT_CUDA_ARCHS must contain at least one CUDA architecture")
if any(not arch.removesuffix("a").isdigit() for arch in cuda_archs):
    raise ValueError(
        "DEEP_SELECT_CUDA_ARCHS must be a comma-separated list such as '90a'"
    )

arch_flags = []
for arch in cuda_archs:
    arch_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

cuda_root = Path(CUDA_HOME)
cuda_targets = [cuda_root / "targets" / target for target in ("x86_64-linux", "sbsa-linux")]
cccl_include_dirs = [target / "include" / "cccl" for target in cuda_targets]
cuda_stub_dirs = [target / "lib" / "stubs" for target in cuda_targets]

extension = CUDAExtension(
    name="deep_select.deep_select_cuda",
    sources=sources,
    include_dirs=[
        str(ROOT / "csrc"),
        str(ROOT / "csrc" / "3rdparty" / "cutlass" / "include"),
        str(ROOT / "csrc" / "3rdparty" / "kerutils" / "include"),
        *(str(path) for path in cccl_include_dirs if path.is_dir()),
    ],
    extra_compile_args={
        "cxx": [
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            "-DKERUTILS_IS_BUILD_ON_CUDA",
        ],
        "nvcc": [
            "-O3",
            "-std=c++20",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ftz=false",
            "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-double-precision-use",
            "-lineinfo",
            "--source-in-ptx",
            *arch_flags,
        ],
    },
    extra_link_args=[
        *(f"-L{path}" for path in cuda_stub_dirs if path.is_dir()),
        "-lcuda",
    ],
)

setup(
    name="deep_select",
    version=VERSION["__version__"],
    packages=find_packages(include=("deep_select", "deep_select.*")),
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
