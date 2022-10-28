__author__ = "Andrea Giovanni Nuzzolese"
__email__ = "andrea.nuzzolese@cnr.it"
__license__ = "Apache 2"
__version__ = "0.2.9"
__status__ = "Alpha"

from pyrml.pyrml import RMLConverter, FunctionAlreadyRegisteredException, RMLFunction
import datetime, locale, hashlib
import uuid, shortuuid
import numpy as np

from typing import List, Dict, Callable, TypeVar

import math

T = TypeVar('T')

'''
Decorator for enabling the registration of function by means of function definition with decoration.
'''
def rml_function(fun_id: str, **params: Dict[str, str]) -> Callable:
    def rml_decorator(f: Callable) -> Callable:
        
        def wrapper(*args, **kwargs) -> T:

            return f(*args, **kwargs)
  
        
        rml_converter = RMLConverter.get_instance()
        if rml_converter.has_registerd_function(fun_id):
            raise FunctionAlreadyRegisteredException(fun_id)
        else:
            rml_f = RMLFunction(fun_id, f, **params)
            rml_converter.register_function(fun_id, rml_f)
    
        
        return wrapper
    
    return rml_decorator


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#toLowerCase', 
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def to_lower_case(value: str) -> str:
    return value.lower()


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#toUpperCase', 
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def to_upper_case(value: str) -> str:
    return value.upper()

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_toNumber', 
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_any_e')
def to_number(value: str) -> float:
    return float(value)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_toTitlecase', 
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def to_titlecase(value: str) -> str:
    value = value.split()
    value = [s[0].upper()+s[1:] for s in value]
    return ' '.join(value)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_trim', 
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def trim(value: str) -> str:
    return value.strip()


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#array_sum', 
              values='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a')
def array_sum(values: List[float]) -> float:
    return sum(values)


@rml_function(fun_id='http://example.com/idlab/function/equal', 
              x='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              y='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter2')
def equal(x: str, y: str) -> bool:
    return x == y


@rml_function(fun_id='http://example.com/idlab/function/notEqual', 
              x='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              y='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter2')
def not_equal(x: str, y: str) -> bool:
    return x != y


@rml_function(fun_id='http://example.com/idlab/function/normalizeDate', 
              date='http://example.com/idlab/function/strDate',
              pattern='http://example.com/idlab/function/pattern')
def normalize_date(date: str, pattern: str) -> str:
    return str(datetime.datetime.strptime(date, pattern).date())


@rml_function(fun_id='http://example.com/idlab/function/normalizeDateTime', 
              date='http://example.com/idlab/function/strDate',
              pattern='http://example.com/idlab/function/pattern')
def normalize_date_time(date: str, pattern: str) -> str:
    return str(datetime.datetime.strptime(date, pattern))

@rml_function(fun_id='http://example.com/idlab/function/normalizeDateTimeWithLang', 
              date='http://example.com/idlab/function/strDate',
              pattern='http://example.com/idlab/function/pattern',
              lang='http://example.com/idlab/function/lang')
def normalize_date_time_with_lang(date: str, pattern: str, lang: str) -> str:
    
    saved_locale = locale.getlocale(locale.LC_TIME)
    locale.setlocale(locale.LC_TIME, lang) 
    
    date_time_str = normalize_date_time(date, pattern)
    
    locale.setlocale(locale.LC_TIME, saved_locale)
    
    return date_time_str


@rml_function(fun_id='http://example.com/idlab/function/normalizeDateWithLang', 
              date='http://example.com/idlab/function/strDate',
              pattern='http://example.com/idlab/function/pattern',
              lang='http://example.com/idlab/function/lang')
def normalize_date_with_lang(date: str, pattern: str, lang: str) -> str:
    
    saved_locale = locale.getlocale(locale.LC_TIME)
    locale.setlocale(locale.LC_TIME, lang) 
    
    date_str = normalize_date(date, pattern)
    
    locale.setlocale(locale.LC_TIME, saved_locale)
    
    return date_str


@rml_function(fun_id='http://example.com/idlab/function/isNull', 
              value='http://example.com/idlab/function/str')
def is_null(value: str) -> bool:
    
    return False if value else True


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#boolean_and', 
              values='http://users.ugent.be/~bjdmeest/function/grel.ttl#param_rep_b')
def boolean_and(values: List[bool]) -> str:
    
    values = np.array(values)
    return np.all(values)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#boolean_or', 
              values='http://users.ugent.be/~bjdmeest/function/grel.ttl#param_rep_b')
def boolean_or(values: List[bool]) -> str:
    
    values = np.array(values)
    return np.any(values)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_min', 
              x='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n',
              y='http://users.ugent.be/~bjdmeest/function/grel.ttl#param_n2',)
def math_min(x: float, y: float) -> float:
    
    return x if x < y else y

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_max', 
              x='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n',
              y='http://users.ugent.be/~bjdmeest/function/grel.ttl#param_n2',)
def math_max(x: float, y: float) -> float:
    
    return x if x > y else y

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#array_length', 
              a='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a')
def array_length(a: List[T]) -> int:
    
    return len(a)
    
@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_length', 
              s='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def string_length(s: str) -> int:
    
    return len(s)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#controls_if', 
              cond='http://users.ugent.be/~bjdmeest/function/grel.ttl#bool_b',
              e_true='http://users.ugent.be/~bjdmeest/function/grel.ttl#any_true',
              e_false='http://users.ugent.be/~bjdmeest/function/grel.ttl#any_false')
def controls_if(cond: bool, e_true: T, e_false: T = None) -> int:
    return e_true if cond else e_false

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#listContainsElement', 
              l='http://example.com/idlab/function/list',
              value='http://example.com/idlab/function/str')
def list_contains_element(l: List[T], value: T) -> bool:
    return value in l

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_contains', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              substring='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_sub')
def string_contains(string: str, substring: str) -> bool:
    return substring in str


@rml_function(fun_id='http://example.com/idlab/function/concat', 
              string1='http://example.com/idlab/function/str',
              string2='http://example.com/idlab/function/otherStr',
              delimiter='http://example.com/idlab/function/delimiter')
def concat(string1: str, string2: str, delimiter: str = '') -> str:
    return delimiter.join([string1, string2])

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replace', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              match='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_find',
              replace='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_replace')
def string_replace(string: str, match: str, replace: str) -> str:
    return string.replace(match, replace)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_replaceChars', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              match='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_find',
              replace='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_replace')
def string_replaceChars(string: str, match: str, replace: str) -> str:
    return string_replace(string, match, replace)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#array_reverse', 
              arr='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a')
def array_reverse(arr: List[T]) -> List[T]:
    return arr[::-1]

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_chomp', 
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              sep='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep')
def string_chomp(value: str, sep: str = '') -> str:
    return value.replace('\n', sep)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#other_coalesce', 
              exprs='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_any_rep_e')
def coalesce(exprs: List[T]) -> T:
    
    for expr in exprs:
        if expr:
            return expr
    
    return None


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_endsWith', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              end='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_sub')
def ends_with(string: str, end: str) -> bool:
    return string.endswith(end)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_startsWith', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              start='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_sub')
def starts_with(string: str, start: str) -> bool:
    return string.startswith(start)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_indexOf', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              substring='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_sub')
def index_of(string: str, substring: str) -> int:
    return string.find(substring)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_lastIndexOf', 
              string='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter',
              substring='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_sub')
def last_index_of(string: str, substring: str) -> int:
    return string.rfind(substring)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#array_join', 
              arr='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
              separator='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_string_sep')
def array_join(arr: List[str], separator: str = '') -> str:
    return separator.join(arr)

@rml_function(fun_id='http://example.com/idlab/function/inRange', 
              test='http://example.com/idlab/function/p_test',
              p_from='http://example.com/idlab/function/p_from',
              p_to='http://example.com/idlab/function/p_to')
def in_range(test: float, p_from: float, p_to: float) -> bool:
    return test in range(p_from, p_to)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_exp', 
              num='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n')
def math_exp(num: float) -> float:
    return math.exp(num)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_floor', 
              num='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n')
def math_floor(num: float) -> float:
    return math.floor(num)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_round', 
              num='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n')
def math_round(num: float) -> float:
    return round(num)

@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_ln', 
              num='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n')
def math_ln(num: float) -> float:
    return math.log(num)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_log', 
              num='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n')
def math_log(num: float) -> float:
    return math.log(num, 10)


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#boolean_not', 
              bool_value='http://users.ugent.be/~bjdmeest/function/grel.ttl#bool_b')
def boolean_not(bool_value: bool) -> bool:
    return not bool_value

@rml_function(fun_id='http://example.com/idlab/function/random')
def random() -> str:
    return str(uuid.uuid4())


@rml_function(fun_id='https://w3id.org/stlab/rml-functions.ttl#short_uuid',
              string='https://w3id.org/stlab/rml-functions.ttl#in_string',
              uuid_len='https://w3id.org/stlab/rml-functions.ttl#uuid_len')
def short_uuid(string: str, uuid_len: int = 8) -> str:
    return shortuuid.uuid(string)[:uuid_len]


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_md5',
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def string_md5(value: str) -> str:
    digest = hashlib.md5(value.encode())
    return digest.hexdigest()


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#string_sha1',
              value='http://users.ugent.be/~bjdmeest/function/grel.ttl#valueParameter')
def string_sha1(value: str) -> str:
    digest = hashlib.sha1(value.encode())
    return digest.hexdigest()


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#array_slice',
              arr='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_array_a',
              from_i='http://users.ugent.be/~bjdmeest/function/grel.ttl#param_int_i_from',
              to_i='http://users.ugent.be/~bjdmeest/function/grel.ttl#param_int_i_opt_to')
def array_slice(arr: List[T], from_i: int = None, to_i: int = None) -> List[T]:
    return arr[from_i:to_i]


@rml_function(fun_id='http://users.ugent.be/~bjdmeest/function/grel.ttl#math_ceil',
              num='http://users.ugent.be/~bjdmeest/function/grel.ttl#p_dec_n')
def math_ceil(num: float) -> float:
    return math.ceil(num)
