import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='iflearn',
     version='0.1.0',
     author="Alicia Curth",
     author_email="aliciacurth@gmail.com",
     description="Machine learning using influence functions",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/AliciaCurth/IF-learn",
     packages=['iflearn'],
     classifiers=[
         "Programming Language :: Python :: 3",
     ],
 )