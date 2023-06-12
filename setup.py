from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='RNN Music Generator',
      version="0.0.1",
      description="RNN Music Generator (api_pred)",
      author="Ariele Amorim, Cristian, Miguel",
      author_email="ariele-plra@hotmail.com",
      #url="",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
