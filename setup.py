#from distutils.core import setup
from setuptools import setup, find_packages

install_requires=[
   'lark-parser>=0.12.0,<=0.12.0',
   'pandas>=2.1.4,<=2.1.4',
   'Jinja2>=3.1.2,<=3.1.2',
   #'rdflib>=6.2.0,<=6.2.0',
   'rdflib>=7.1.3,<=7.1.3',
   'SPARQLWrapper>=2.0.0,<=2.0.0',
   #'Unidecode>=1.3.6,<=1.3.6',
   'Flask>=2.2.2,<=2.2.2',
   'jsonpath-ng>=1.5.3,<=1.5.3',
   'shortuuid>=1.0.9,<=1.0.9',
   'numpy>=1.26.4,<=1.26.4',
   'python-slugify[unidecode]>=7.0.0,<=7.0.0',
   'lxml>=5.1.0,<=5.1.0',
   'SQLAlchemy>=1.4.46,<=1.4.46',
   'psycopg2>=2.9.10,<=2.9.10',
   'mysql-connector-python>=9.2.0,<=9.2.0',
   'pymssql>=2.3.2,<=2.3.2'
]

setup(name='pyrml', version='0.4.0',
    packages=find_packages(), package_data={'pyrml': ['grammar.lark']}, install_requires=install_requires)
