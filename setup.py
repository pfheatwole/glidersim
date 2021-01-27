from setuptools import find_namespace_packages, setup


if __name__ == "__main__":
    setup(
        name="pfh.glidersim",
        url="https://github.com/pfheatwole/glidersim",
        license="MIT",
        package_dir={"": "src"},
        packages=find_namespace_packages(where="src"),
        package_data={
            "pfh.glidersim.extras": [
                "data/*/*.csv",
                "data/*/README",
            ],
        },
        use_scm_version=True,
        python_requires="~=3.7",
        setup_requires=["setuptools_scm"],
        install_requires=[
            "numpy >= 1.19.4",
            "scipy >= 1.6.0",
            "numba >= 0.52.0",
            "matplotlib >= 3.3.3",
        ],
        extras_require={
            "dev": [
                "flake8 >= 3.8.4",
                "flake8-black >= 0.2.1",
                "flake8-bugbear >= 20.11.1",
                "flake8-commas >= 2.0.0",
                "flake8-docstrings >= 1.5.0",
                "flake8-isort >= 3.0.0",
                "flake8-rst-docstrings >= 0.0.14",
                "pre-commit >= 2.10.0",
                "Sphinx >= 3.4.3",
                "numpydoc >= 1.1.0",
                "sphinx-rtd-theme >= 0.5.1",
            ],
        },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Physics",
        ],
    )
