#from distutils.core import setup
from setuptools import setup, find_packages

install_requires=[
   'lark-parser>=0.8.5,<=0.8.5',
   'pandas',
   'Jinja2>=2.11.3,<=2.11.3',
   'rdflib>=6.1.1,<=6.1.1',
   'Unidecode>=1.1.1,<=1.1.1',
   'Flask>=1.0,<=1.0',
   'jsonpath-ng>=1.5.2,<=1.5.2'
]

setup(name='pyrml', version='0.2.9',
    packages=find_packages(), package_data={'pyrml': ['grammar.lark']}, install_requires=install_requires)
