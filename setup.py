#from distutils.core import setup
from setuptools import setup, find_packages

install_requires=[
   'lark-parser>=0.12.0,<=0.12.0',
   'pandas>=1.5.1,<=1.5.1',
   'Jinja2>=3.1.2,<=3.1.2',
   'rdflib>=6.2.0,<=6.2.0',
   #'Unidecode>=1.3.6,<=1.3.6',
   'Flask>=2.2.2,<=2.2.2',
   'jsonpath-ng>=1.5.3,<=1.5.3',
   'shortuuid>=1.0.9,<=1.0.9',
   'numpy>=1.23.4,<=1.23.4',
   'python-slugify[unidecode]>=7.0.0,<=7.0.0'
]

setup(name='pyrml', version='0.3.0',
    packages=find_packages(), package_data={'pyrml': ['grammar.lark']}, install_requires=install_requires)
