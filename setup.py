from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fermat',
    version='0.0.3',
    description='library to compute fermat distance',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://bitbucket.org/aristas/fermat',
    license='MIT',
    author='Facundo Sapienza',
    author_email='f.sapienza@aristas.com.ar',
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
