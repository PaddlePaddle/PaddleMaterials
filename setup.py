from pathlib import Path

import numpy as np
import setuptools
from Cython.Build import cythonize
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools import Extension

"""
Setup configuration
"""


extensions = [
    Extension(
        "ppmat.models.mattersim.threebody_indices",
        ["ppmat/models/mattersim/threebody_indices.pyx"],
        include_dirs=[np.get_include()],
    )
]

BUILD_AND_TEST_REQUIREMENTS = {
    "cython",
    "pytest",
    "setuptools-scm",
}


def get_readme() -> str:
    """Read the PyPI package description."""
    return Path("README_PYPI.md").read_text(encoding="utf-8")


def get_requirements() -> list[str]:
    """Read runtime requirements, excluding build and test dependencies."""
    requirements = []
    seen = set()
    # Keep setuptools at runtime while matminer 0.9.2 relies on pkg_resources.
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name in BUILD_AND_TEST_REQUIREMENTS or name in seen:
            continue
        requirements.append(str(requirement))
        seen.add(name)
    return requirements


if __name__ == "__main__":
    setuptools.setup(
        name="ppmat",
        author="PaddlePaddle",
        url="https://github.com/PaddlePaddle/PaddleMaterials",
        license="Apache-2.0",
        description=("An AI-driven materials science toolkit based on PaddlePaddle."),
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_namespace_packages(include=("ppmat", "ppmat.*")),
        package_data={"": ["*.json"]},
        license_files=("LICENSE",),
        python_requires=">=3.10",
        project_urls={
            "Documentation": "https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/Install.md",
            "Issues": "https://github.com/PaddlePaddle/PaddleMaterials/issues",
            "Source": "https://github.com/PaddlePaddle/PaddleMaterials",
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        install_requires=get_requirements(),
        use_scm_version=True,
        ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
        zip_safe=False,
    )
