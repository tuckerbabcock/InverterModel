from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('invertermodel/__init__.py').read(),
)[0]

setup(name='invertermodel',
      version=__version__,
      author='Tucker Babcock',
      author_email='tuckerbabcock1@gmail.com',
      url='https://github.com/tuckerbabcock/InverterModel',
      license='MPL-2.0 License',
      packages=[
          'invertermodel',
      ],
      python_requires=">=3.10",
      install_requires=[
          'numpy>=1.21.4',
          'openmdao>=3.18.0',
      ],
      classifiers=[
        "Programming Language :: Python"
      ]
)
