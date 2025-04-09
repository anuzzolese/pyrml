from argparse import ArgumentParser
import logging
import os, codecs
from pathlib import Path
from pyrml.pyrml_mapper import RMLConverter
from pyrml.functions import *


class PyrmlCMDTool:

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-o", "--output", dest="output",
                    help="Output file. If no choice is provided then standard output is assumed as default.", metavar="RDF out file")
        parser.add_argument("-f", "--output-format", dest="format",
                    help="Output file format. Possible values are n3, nquads, nt, pretty-xml, trig, trix, turtle, and xml. If no choice is provided then NTRIPLES is assumed as default.", metavar="RDF out file")
        parser.add_argument("-m", action="store_true", default=False,
                            help="Enable conversion based on multiproccessing for fastening the computation.")
        parser.add_argument("input", help="The input RML mapping file for enabling RDF conversion.")
        

        self.__args = parser.parse_args()
        
        logging.basicConfig(level=logging.DEBUG)
        
    def do_map(self):
        rml_converter =Framework.get_mapper()

        #Inizio aggiunta per recogito
        #rml_converter.register_function("get_id", get_id)
        #rml_converter.register_function("get_uri", get_uri)
        #Fine aggiunta per recogito
        
        
        g = rml_converter.convert(self.__args.input, self.__args.m)
        
        if self.__args.format is not None:
            format = self.__args.format
        else:
            format = 'nt'
        
        if self.__args.output is not None:
            dest_folder = Path(self.__args.output).parent
            
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
                
            with codecs.open(self.__args.output, 'w', encoding='utf8') as out_file:
                out_file.write(g.serialize(format=format))
                
        else:
            logging.info(g.serialize(format=format))
            
              
def get_id(string):
    return string.split(":")[1]

def get_uri(string):
    prefix = string.split(":")[0]
    id = string.split(":")[1]
    prefix_map = {"l0":"https://w3id.org/italia/onto/l0/"}
    return prefix_map[prefix]+id

if __name__ == '__main__':
    
    PyrmlCMDTool().do_map()
