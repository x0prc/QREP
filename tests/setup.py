from setuptools import setup, find_packages

setup(
    name="QREP",
    version="0.0.1",
    description="Quantum-Resistant Privacy Engine with Differential Privacy and Federated Learning",
    author="x0prc",
    url="https://github.com/x0prc/QREP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8, <4",
    install_requires=[
        "cryptography>=42.0.7",
        "pqcrypto>=0.1.3",
        "pynput>=1.7.6",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.38.2",
        "stylegan2_pytorch",
        "numpy>=1.26.0",
        "pandas>=2.1.3",
        "scikit-learn>=1.3.2",
        "Pillow>=10.1.0",
        "syft>=0.8.0",
        "tqdm>=4.66.1",
        "requests>=2.31.0",
        "pytest>=8.0.2"
    ],
    entry_points={
        "console_scripts": [
            "qrpe-biometrics=tokenization.biometric_capture:main",
            "qrpe-train-gan=differential_privacy.gan_manager:main",        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
