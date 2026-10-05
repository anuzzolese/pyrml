from setuptools import setup, find_packages

install_requires=[
   'lark-parser>=0.12.0,<1',
   'pandas>=2.1.4,<3',
   'Jinja2>=3.1.2,<4',
   'rdflib>=7.1.3,<8',
   'SPARQLWrapper>=2.0.0,<3',
   'jsonpath-ng>=1.5.3,<2',
   'shortuuid>=1.0.9,<2',
   'numpy>=1.26.4',
   'python-slugify[unidecode]>=7.0.0',
   'lxml>=5.1.0',
   'SQLAlchemy>=2.0'
]

setup(name='pyrml-lib', version='0.6.0',
    packages=find_packages(), package_data={'pyrml': ['grammar.lark']}, install_requires=install_requires,
    python_requires='>=3.9')
