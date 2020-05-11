from distutils.core import setup

install_requires=[
   'lark-parser>=0.8.5,<=0.8.5',
   'pandas>=0.24.1,<=0.24.1',
   'Jinja2>=2.10,<=2.10',
   'rdflib>=4.2.2,<=4.2.2',
   'rdflib>=4.2.2,<=4.2.2',
   'Unidecode>=1.1.1,<=1.1.1'
]

setup(name='pyrml', version='1.0.0', install_requires=install_requires)

