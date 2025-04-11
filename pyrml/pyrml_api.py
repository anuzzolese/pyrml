__author__ = "Andrea Giovanni Nuzzolese"
__email__ = "andrea.nuzzolese@cnr.it"
__license__ = "Apache 2"
__version__ = "0.2.9"
__status__ = "Alpha"


from abc import ABC, abstractmethod
import hashlib
import os
import re
from typing import Set, Dict, Type

from lark import Lark, Token
from lark.visitors import Transformer
from rdflib import URIRef, Graph, BNode, Literal, IdentifiedNode
from rdflib.term import Node
import unidecode
import pandas as pd
import numpy as np
from urllib.parse import quote
from rdflib.plugin import register, Parser

class FunctionNotRegisteredException(Exception):
    def __init__(self, function, message="The function {0} does not exist in the function registry"):
        self.function = function
        self.message = message.format(function)
        super().__init__(self.message)
        
class NoneFunctionException(Exception):
    def __init__(self, message="NoneType is not a valid callable object. Hence, no function can be retrieved from the fucntion registry of the RMLConverter."):
        self.message = message
        super().__init__(self.message)
        
class FunctionAlreadyRegisteredException(Exception):
    def __init__(self, function, message="A function with the ID {0} already exists in the function registry of RMLConverter. If you want to update the registry please unregister the existing one before registering a new one."):
        self.message = message.format(function)
        super().__init__(self.message)
        
class ParameterNotExintingInFunctionException(Exception):
    def __init__(self, function, parameter, message="The parameter {1} referred to in the function {0} was never declared as one of the arguments of such a function."):
        self.message = message.format(function.__name__, parameter)
        super().__init__(self.message)


def graph_add_all(g1, g2):
    for (s,p,o) in g2:
        g1.add((s,p,o))
    return g1


def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

class TermMap(ABC):
    
    def __init__(self, map_id: IdentifiedNode = None):
        if map_id is None:
            self._id = BNode()
        else:
            self._id = map_id
            
        self._function_map = None
        
    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        if isinstance(other, TermMap):
            return self._id == other._id
        return NotImplemented
        
    @property
    def function_map(self):
        return self._function_map
         
    
    @property        
    def id(self) -> IdentifiedNode:
        return self._id
    
    @abstractmethod
    def get_mapped_entity(self) -> Node:
        pass
    
    @abstractmethod
    @multigen
    def apply(self, row: pd.Series = None) -> object:
        pass
    
    @staticmethod
    @abstractmethod
    def from_rdf(g: Graph) -> Set[object]:
        pass
    
class Evaluable():
    
    @abstractmethod
    def eval(self, row, is_iri):
        pass
    
    @abstractmethod
    def eval_(self, row, is_iri):
        pass

class Funz(Evaluable):
    
    def __init__(self, fun, args):
        self.__fun = fun
        self.__args = args
        
    
    def eval(self, row, is_iri):
        args = []
        for arg in self.__args:
            if isinstance(arg, str) and arg.strip() == '*':
                args.append(row)
            elif isinstance(arg, str):
                args.append(TermUtils.replace_place_holders(arg, row, False))
            else:
                args.append(arg)
        value = self.__fun(*args)
        
        return TermUtils.irify(value) if is_iri else value
    
    def eval_(self, row, is_iri):
        args = []
        for arg in self.__args:
            if isinstance(arg, str) and arg.strip() == '*':
                args.append(row)
            elif isinstance(arg, str):
                args.append(TermUtils.replace_place_holders(arg, row, False))
            else:
                args.append(arg)
        value = self.__fun(*args)
        
        return TermUtils.irify(value) if is_iri else value
    
    def _eval_(self, row, columns, is_iri):
        args = []
        for arg in self.__args:
            argval = arg.value if isinstance(arg, Token) else arg
            if isinstance(argval, str) and argval.strip() == '*':
                _input = {col:row[val] for col, val in columns.items()}
                args.append(_input)
            elif isinstance(argval, str):
                args.append(TermUtils.replace_place_holders_(argval, row, columns, False))
            else:
                args.append(argval)
        value = self.__fun(*args)
        
        return TermUtils.irify(value) if is_iri else value
    
class String(Evaluable):
    def __init__(self, string):
        self.__string = string
    
    def eval(self, row, is_iri):
        return TermUtils.replace_place_holders(self.__string, row, is_iri)
    
    def eval_(self, row, is_iri):
        return TermUtils.replace_place_holders(self.__string, row, is_iri)
    
    def _eval_(self, row, columns, is_iri):
        return TermUtils.replace_place_holders_(self.__string, row, columns, is_iri)
    
    def __str__(self):
        return self.__string
    
class Expression():
    
    def __init__(self):
        self._subexprs = []
        
    def add(self, subexpr: Evaluable):
        self._subexprs.append(subexpr)
        
    def eval(self, row, is_iri):
        items = [item.eval_(row, is_iri) for item in self._subexprs]
        try:
            value = "".join(items)
        except:
            print(f'1: {self._subexprs}')
            print(f'2: {[str(item) for item in self._subexprs]}')
            print(f'3: {row}')
            for item in self._subexprs:
                print(f'\t {item.eval(row, is_iri)}')
            raise
        
        if value != '':
            return URIRef(value) if is_iri else value
        else:
            return None
        
    def _eval_(self, row, columns, is_iri):
        
        items = [item._eval_(row, columns, is_iri) for item in self._subexprs]
        try:
            value = "".join(items)
        except:
            '''
            print(f'1: {self._subexprs}')
            print(f'2: {[str(item) for item in self._subexprs]}')
            print(f'3: {row}')
            for item in self._subexprs:
                print(f'\t {item.eval(row, is_iri)}')
            raise
            '''
            value = None
        
        if value:
            return URIRef(value) if is_iri else value
        else:
            return None
        
    def eval_(self, data_source: 'DataSource', is_iri: bool):
        
        #vect_eval = np.vectorize(self.eval)
        
        #return vect_eval(data_source.data, is_iri)
        
        return [self._eval_(row, data_source.columns, is_iri) for row in data_source.data]
        
        
        
    @classmethod
    def create(cls, value: Literal) -> 'Expression':
        _expr = Expression()
        if value:
            value_str = value.value
            p = re.compile('(?<=\%eval:).+?(?=\%)')
            
            matches = p.finditer(value_str)
            #s = "'{mapped_entity}'".format(mapped_entity=mapped_entity.replace("'", "\\'"))
            s = value_str
            
            cursor = 0

            for match in matches:
                
                start = match.span()[0]-6
                end = match.span()[1]+1
                
                if cursor < start:
                    _expr.add(String(s[cursor:start]))
                    
                function = match.group(0)
                
                result = TermUtils.get_functions(function)
                
                _expr.add(Funz(result[0], result[1]))
                
                cursor = end
                
            if cursor < len(s):
                _expr.add(String(s[cursor:]))
        
        return _expr
        
    
class AbstractMap(TermMap):
    def __init__(self, map_id: URIRef = None, mapped_entity: Node = None):
        super().__init__(map_id)
        self._mapped_entity = mapped_entity
        
        self._expression = Expression()
        
        if mapped_entity is not None and isinstance(mapped_entity, str):
            p = re.compile('(?<=\%eval:).+?(?=\%)')
            
            matches = p.finditer(mapped_entity)
            #s = "'{mapped_entity}'".format(mapped_entity=mapped_entity.replace("'", "\\'"))
            s = mapped_entity
            
            cursor = 0

            #test = "Ciccio b'ello"
            #test = "\"{t}\"".format(t=test)
            out = ''
            #print(eval(repr(test)))
            for match in matches:
                
                start = match.span()[0]-6
                end = match.span()[1]+1
                
                if cursor < start:
                    self._expression.add(String(s[cursor:start]))
                    
                #print("%d, %d"%(start, end))
                function = match.group(0)
                #text = "%eval:" + function + "%"
                
                #function = TermUtils.replace_place_holders(function, row, False)
                result = TermUtils.get_functions(function)
                
                self._expression.add(Funz(result[0], result[1]))
                
                #print(result[0], *result[1])
                #result = "{fun}{params}".format(fun=result[0], params=tuple(result[1]))
                #print(result)
                
                #out += '+' + result
                
                cursor = end
                
            if cursor < len(s):
                self._expression.add(String(s[cursor:]))
                #out += '+' + s[cursor:] 
            
            
            #value = TermUtils.replace_place_holders(s, row, is_iri)
            
    def get_mapped_entity(self) -> Node:
        return self._mapped_entity
    
    '''
    @staticmethod
    def get_rml_converter():
        from pyrml.pyrml_mapper import RMLConverter
        return RMLConverter.get_instance()
    '''
    
class TermUtils():
    
    @staticmethod
    def digest(s):
        hash = hashlib.md5(s.encode())
        return hash.hexdigest()
    
    @staticmethod
    def is_valid_language_tag(tag):
        regex = re.compile(r'^((?:(en-GB-oed|i-ami|i-bnn|i-default|i-enochian|i-hak|i-klingon|i-lux|i-mingo|i-navajo|i-pwn|i-tao|i-tay|i-tsu|sgn-BE-FR|sgn-BE-NL|sgn-CH-DE)|(art-lojban|cel-gaulish|no-bok|no-nyn|zh-guoyu|zh-hakka|zh-min|zh-min-nan|zh-xiang))|((?:([A-Za-z]{2,3}(-(?:[A-Za-z]{3}(-[A-Za-z]{3}){0,2}))?)|[A-Za-z]{4})(-(?:[A-Za-z]{4}))?(-(?:[A-Za-z]{2}|[0-9]{3}))?(-(?:[A-Za-z0-9]{5,8}|[0-9][A-Za-z0-9]{3}))*(-(?:[0-9A-WY-Za-wy-z](-[A-Za-z0-9]{2,8})+))*(-(?:x(-[A-Za-z0-9]{1,8})+))?)|(?:x(-[A-Za-z0-9]{1,8})+))$', re.IGNORECASE)
        return True if re.match(regex, tag) else False
        
    
    @staticmethod
    def urify(entity, row):
        if isinstance(entity, Literal):
            s = TermUtils.eval_functions(entity, row, True)
            if s is not None and s.strip() != '':
                return URIRef(s)
            else:
                return float('nan')
        elif isinstance(entity, URIRef):
            return entity
        
    @staticmethod
    def replace_place_holders(value, row, is_iri):
        #p = re.compile('\{(.+)\/?\}')
        #p = re.compile('(?<=\{).+?(?=\})')
        p = re.compile('(?<!\\)\{.+?(?<!\\)\}')
        
        row = row._asdict()
        
        matches = p.finditer(value)
        
        #input_value = value
        s = value
        
        for match in matches:
            column = match.group(0)[1:-2]
            
            column_key = column.strip().replace(r' ', '_')
            if column_key in row:
                text = "{( )*" + re.escape(column) + "( )*}"
                
                if row[column_key] != row[column_key]:
                    s = re.sub(text, '', s)
                else:
                    #print(row.index)
                    if column not in row:
                        column += "_l"
                    
                    cell_value = str(row[column_key])
                    '''
                    if span_start>0 and span_end<len(input_value):
                        if input_value[span_start] == '\'' and input_value[span_end] == '\'':
                            cell_value = cell_value.replace('\'', '\\\\\'')
                        elif input_value[span_start] == '"' and input_value[span_end] == '"':
                            cell_value = cell_value.replace('"', '\\\"')
                    '''
                    
                    if is_iri:
                        value = quote(cell_value, safe='')
                    else:
                        value = cell_value
                    s = re.sub(text, value, s)
            else:
                return None
            #print(str(row[column]))
            
        
        return s
    
    
    @staticmethod
    def replace_place_holders_(value, row, columns, is_iri):
        #p = re.compile('(?<=\{).+?(?=\})')
        p = re.compile(r'(?<!\\)\{.+?(?<!\\)\}')
        
        matches = p.finditer(value)
        
        #input_value = value
        s = value
        
        for match in matches:
            column = match.group(0)[1:-1]
            
            #column_key = column.strip().replace(r' ', '_')
            column_key = column.strip("' ")
            
            if column_key not in columns:
                column_key = f'{column_key}_r'
            
            if column_key not in columns:
                column_key = column_key.lower()
            
            if column_key not in columns:
                column_key = column_key.upper() 
            
            if column_key in columns:
                
                text = "{( )*" + re.escape(column) + "( )*}"
                target_value = DataSource.get_from_row(row, columns, column_key)
                
                if target_value != target_value:
                    #s = re.sub(text, '', s)
                    return None
                else:
                    target_value = DataSource.get_from_row(row, columns, column_key)
                    
                    if target_value is None or (isinstance(target_value, float) and np.isnan(target_value)):
                        s = None
                    else:
                        target_value = str(target_value)
                        if is_iri:
                            value = quote(target_value, safe='')
                        else:
                            value = target_value
                            
                        s = re.sub(text, value, s)
                        s = re.sub('\\\\', '', s)
            else:
                return None
            #print(str(row[column]))
            
        
        return s
        
    @staticmethod
    def __eval_functions(text, row=None):
        
        return EvalParser.parse(text, row)
    
    @staticmethod
    def get_functions(text, row=None):
        
        return EvalParser.parse(text, row)
    
    @staticmethod
    def eval_functions(value, row, is_iri):
        #p = re.compile('\{(.+)\/?\}')
        
        if value is not None:
            p = re.compile('(?<=\%eval:).+?(?=\%)')
        
            matches = p.finditer(value)
            s = value

            for match in matches:
                function = match.group(0)
                text = "%eval:" + function + "%"
                
                function = TermUtils.replace_place_holders(function, row, False)
                
                result = TermUtils.__eval_functions(function, row)
                
                if result is None:
                    result = ""
                s = s.replace(text, result)
            
            value = TermUtils.replace_place_holders(s, row, is_iri)
            
        return value
    
    
    @staticmethod
    def eval_template(template, row, is_iri):
        s = TermUtils.replace_place_holders(template, row, is_iri)
        s = eval(repr(s))
        #print(s)
        return s
        
    @staticmethod
    def irify(string):
        
        if string and isinstance(string, str) and PyRML.IRIFY:
            
            '''
            The followint regex pattern allows to check if the input string is provided as a valid URI. E.g. http://dati.isprambiente.it/rmn/Ancona.jpg
            '''
            regex = re.compile(
                r'^(?:http|ftp)s?://' # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
                r'localhost|' #localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
                r'(?::\d+)?' # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
            
            '''
            In case the input sstring is not a valid URI than the function applies the irification (i.e. the transormation aimed at removing characters that prevent
            an IRI to be valid).
            '''
            
            if re.match(regex, string) is None:
        
                string = unidecode.unidecode(string)
                string = string.lower();
                string = re.sub(r'[\']', '', string)
                #string = re.sub(r'[;.,&"???!]', '', string)
                string = re.sub(r'[;,&"???!]', '', string)
                string = re.sub(r'[ \/]', '_', string);
                string = re.sub(r'[\(\)]', '', string);
                string = re.sub(r'\-$', '', string);
                string = re.sub(r'(\-)+', '_', string);
                string = re.sub(r'(\_)+', '_', string);
                string = re.sub(r'(\|)+', '_', string);
                
            string = re.sub(r'(<)', '%3C', string);
            string = re.sub(r'(>)', '%3E', string);
        return string
    
class EvalParser():
    
    dirname = os.path.dirname(__file__)
    lark_grammar_file = os.path.join(dirname, 'grammar.lark')
    LARK = Lark.open(lark_grammar_file,parser='lalr')
    
    @staticmethod
    def parse(expr, row=None):
        #logging.debug("Expr", expr)
        tree = EvalParser.LARK.parse(expr)
        return EvalTransformer(row).transform(tree)
    
class EvalTransformer(Transformer):
    
    def __init__(self, row=None):
        self.__row = row
    
    def start(self, fun):
        return fun
        #return "%s(%s)"(fun[0],*fun[1])
    
    def f_name(self, name):
        #rml_converter = AbstractMap.get_rml_converter().get_instance()
        if PyRML.has_registerd_function(name[0]):
            fun = PyRML.get_registerd_function(name[0])
            name[0] = fun.__qualname__
            
            return fun
            
        return None
    
    def parameters(self, parameters):
        return parameters
    
    def paramvalue(self, param):
        return param[0]
    
    def row(self, val):
        return '*'
    
    def string(self, val):
        return val[0][1:-1]
    
    def placeholder(self, val):
        return val[0][:]
    
    def number(self, val):
        return val[0]
    
    def dec_number(self, val):
        return int(val[0])
    
    def hex_number(self, val):
        return hex(val[0])
    
    def bin_number(self, val):
        return bin(val[0])
    
    def oct_number(self, val):
        return oct(val[0])
    
    def float_number(self, val):
        return float(val[0])
    
    def imag_number(self, val):
        return complex(val[0])
    
    def const_true(self, val):
        return True
    
    def const_false(self, val):
        return False
    
    def const_none(self, val):
        return None
    
    
class MappingsDict():
    
    def __init__(self):
        self.__dict = dict()
        MappingsDict.__instance = self
    
    def __iter__(self):
        return self.__dict.__iter__()
    
    def __next__(self):
        return self.__dict.__next__()
    
    def add(self, term_map : TermMap):
        if isinstance(term_map.id, URIRef):
            self.__dict.update( {term_map.id: term_map} )
            
    def get(self, iri : URIRef):
        return self.__dict[iri]
    


class DataSource():
    
    def __init__(self, df: pd.DataFrame):
        self.__data = df.to_numpy()
        self.__columns = {col: count for count, col in enumerate(df.columns)}
        self.__df = df
        
    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__df
    
    @property
    def data(self) -> np.ndarray:
        return self.__data
    
    @property
    def columns(self) -> dict:
        return self.__columns
    
    def get(self, row, column_reference: str) -> object:
        if column_reference in self.columns:
            index = self.columns[column_reference]
            return row[index]
        else:
            return None
        
    @staticmethod
    def get_from_row(row: np.ndarray, columns: dict, column_reference: str) -> object:
        if column_reference not in columns:
            column_reference = column_reference.lower()
            if column_reference not in columns:
                column_reference = column_reference.upper()
                if column_reference not in columns:
                    return None
            
        index = columns[column_reference]
        return row[index]
        
        
class Mapper(ABC):
    
    @abstractmethod
    def convert(self, rml_mapping: str, multiprocessed=False, template_vars: Dict[str, str] = None) -> Graph:
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
class RMLModelException(Exception):
    
    def __init__(self, message):            
        super().__init__(message)
    
class PyRML():
    
    __mapper = None
    __function_registry = dict()
    
    register('text/turtle', Parser, 'pyrml.pyrml_rdflib', 'MyTurtleParser')
    register('turtle', Parser, 'pyrml.pyrml_rdflib', 'MyTurtleParser')
    register('ttl', Parser, 'pyrml.pyrml_rdflib', 'MyTurtleParser')
    
    #IRIFY = True
    #RML_STRICT = False
    IRIFY = False
    RML_STRICT = True
    INFER_LITERAL_DATATYPES = False
    
    @classmethod
    def set_mapper(cls, mapper):
        cls.delete_mapper()
        cls.__mapper = mapper
    
    @classmethod
    def get_mapper(cls) -> Mapper:
        if cls.__mapper is None:
            
            module = __import__('pyrml.pyrml_mapper')
            mapper_cls = getattr(module, 'RMLConverter')
            cls.__mapper = mapper_cls()

        return cls.__mapper
    
    @classmethod
    @property
    def function_registry(cls):
        return cls.__function_registry
    
    @classmethod
    def register_function(cls, name, fun):
        cls.__function_registry.update({name: fun})
        
    @classmethod
    def unregister_function(cls, name):
        del cls.__function_registry[name]
        
    @classmethod
    def has_registerd_function(cls, name):
        return name in cls.__function_registry
    
    @classmethod
    def get_registerd_function(cls, name):
        return cls.__function_registry.get(name)
    
    
    @classmethod
    def delete_mapper(cls):
        
        cls.__mapper.reset()
        cls.__mapper = None
    