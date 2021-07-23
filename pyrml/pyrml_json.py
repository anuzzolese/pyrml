import json
from jsonpath_ng import jsonpath, parse
import pandas as pd

jsonpath_expr = parse('$.venue[*]')

with open('Venue.json') as f:
    json_data = json.load(f)
    
    matches = jsonpath_expr.find(json_data)
    
    data = [match.value for match in matches]
            
df = pd.json_normalize(data)
print(df)