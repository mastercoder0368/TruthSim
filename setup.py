from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="truthsim",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="TruthSim: Truth-Preserving Noisy Patient Simulator for Medical AI Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/truthsim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "truthsim-preprocess=scripts.preprocess_umls:main",
            "truthsim-run=scripts.run_simulation:main",
            "truthsim-evaluate=scripts.evaluate:main",
        ],
    },
)
