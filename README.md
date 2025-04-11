# pyRML
pyRML is a Python based engine for processing RML files. The [RDF Mapping Language](https://rml.io/specs/rml/) (RML) is a mapping language defined to express customized mapping rules from heterogeneous data structures and serializations to the RDF data model. RML is defined as a superset of the W3C-standardized mapping language [R2RML](https://rml.io/specs/rml/#bib-r2rml), aiming to extend its applicability and broaden its scope, adding support for data in other structured formats.
### Installation
pyRML requires Python 3.
To install the engine use the Python package manager `pip`, i.e.:
```
pip install pyrml-lib
```

Alternatively, it is possible to run `pip` under the root directory of source code of PyRML once it has been downloaded. For example:

```
pip install .
```

Alternatively, it is possible to install the pyRML package directly from GitHub in the following way:

```
pip install git+https://github.com/anuzzolese/pyrml
```

### Usage

It is possible to use pyRML either by means of its API or the command line tool that is provided along with the source package.

###### API
The ```Mapper``` is the key class of pyRML. It accepts the path to an RML file as input and returns an RDF graph as output. The output graph is an instance of the class ```Graph``` provided by [RDFLib](https://github.com/RDFLib/rdflib).
```python
from pyrml import Mapper, PyRML
from rdflib import Graph
import os

# Create an instance of RML Mapper with PyRML.
mapper : Mapper = PyRML.get_mapper()

'''
Invoke the method convert on the instance of class RMLConverter by:
 - using the file examples/artist/artist-map.ttl (see the examples in this repo);
 - obtaining an RDF graph as output.
'''

rml_file_path = os.path.join('examples', 'artists', 'artist-map.ttl')
rdf_graph : Graph  = mapper.convert(rml_file_path)

# Print the triples contained into the RDF graph.
for s,p,o in rdf_graph:
    print(s, p, o)
```

###### Command line tool
The command line tool is implemented by the script ```pyrml-mapper.py```.
Such a script can be used in the following way:

```bash
python pyrml-mapper.py [-o RDF out file] [-f RDF out file] [-m] input
```

where:
 - the positional argument ```input``` is the input RML mapping file for enabling the RDF conversion;
 - the optional argument ```-o filename``` is the file to store the resulting RDF graph. If no choice is provided then standard output is assumed as default.
 - the optional argument ```-f rdf-syntax``` can be used to specify the syntax to serialize the RDF graph. Possible values are n3, nquads, nt, pretty-xml, trig, trix, turtle, and xml. If no choice is provided then NTRIPLES is assumed as default.
 - the optional flag ```-m``` enables the conversion based on multiprocessing for speeding up the transformation process.
 
The following is an example about how to use the command line tool for processing the RML file available in ```examples/artists/artist-map.ttl```, thus converting the CSV files ```examples/artists/Artist.csv``` and ```examples/artists/Place.csv``` into an RDF graph serialized as TURTLE and stored into the file named ```artists_places.ttl```.

```bash
python pyrml-mapper.py -o artists_places.ttl -f turtle examples/artists/artist-map.ttl
```