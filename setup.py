from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()
    
setup(
    name="voxlect",
    version="0.1.0",
    packages=find_packages(),  # auto-detects packages in the folder
    install_requires=required,
    author="Tiantian Feng",
    author_email="tiantiaf@usc.edu",
    description="This repo includes Voxlect, one of the first benchmarking efforts that predict dialects and regional languages worldwide using speech foundation models.",
    url="https://github.com/tiantiaf0627/vox-profile-release",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)