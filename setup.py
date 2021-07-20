from setuptools import find_packages, setup

setup(
    name="model",
    packages=find_packages(),
    version="0.1.0",
    description="Face recognition contest",
    author="Andrei Belenko",
    install_requires=[
        "torch == 1.9.0",
        "torchvision == 0.10.0",
        "opencv-python == 4.5.3.56",
        "pandas == 1.2.4",
        "scikit-learn == 0.24.2",
        "notebook == 6.4.0",
        "matplotlib == 3.4.2",
        "tqdm == 4.61.2"
    ]
)
