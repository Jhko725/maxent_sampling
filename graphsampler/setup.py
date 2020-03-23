import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphsampler", # Replace with your own username
    version="1.0.0",
    author="Joon-Hyuk Ko, Sung Hoon Kim",
    author_email="jhko725@snu.ac.kr, josephkim11@snu.ac.kr",
    description="Final Project for Advanced Data Mining (Fall 2019), SNU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    install_requires = ['numpy', 'snap-stanford', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.5',
)