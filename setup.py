from setuptools import find_namespace_packages, setup


if __name__ == "__main__":
    setup(
        name="pfh.glidersim",
        url="https://github.com/pfheatwole/glidersim",
        license="MIT",
        package_dir={"": "src"},
        packages=find_namespace_packages(where="src"),
        include_package_data=True,
        python_requires=">=3.8",
        install_requires=[
            "numpy >= 1.20",
            "scipy",
            "numba >= 0.54.0",  # Supports numpy v1.20
            "matplotlib",
        ],
        extras_require={
            "dev": [
                "flake8",
                "flake8-black",
                "flake8-bugbear",
                "flake8-commas",
                "flake8-docstrings",
                "flake8-isort",
                "flake8-rst-docstrings",
                "isort",
                "pre-commit",
                "mypy",
                "Sphinx",
                "sphinx-rtd-theme",
                "numpydoc",
            ],
        },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Physics",
        ],
    )
