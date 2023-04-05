from pyrml.pyrml_mapper import RMLConverter
import os

path = '/Users/andrea/Downloads/ISWC 2022_data_EasyChair_2023-04-04-CSVFilesForMetadata'

# Create an instance of the class RMLConverter.
rml_converter = RMLConverter()

rml_file_path = os.path.join(path, 'sd-rml-mapping.ttl')

data = {
		'SUBMISSIONS': os.path.join(path, 'submission.csv'),
		'TRACKS': os.path.join(path, 'track.csv'),
		'CONFERENCE': 'iswc',
		'CONFERENCE_UP': 'ISWC',
		'YEAR': '2022',
		'START_DATE': '2022-10-23T09:00:00',
		'END_DATE': '2022-10-27T18:00:00'
	}

rdf_graph = rml_converter.convert_(rml_file_path, template_vars=data)


print(f'The output graph contains {len(rdf_graph)} RDF triples')

out_file = os.path.join(path, 'conf.ttl')
rdf_graph.serialize(format='turtle', destination=out_file)
