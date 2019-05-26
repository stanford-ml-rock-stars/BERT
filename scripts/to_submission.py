import csv
import argparse
import re
def is_number(num):
    pattern = re.compile(r'^([-+]?[0-9]\d*\.\d*|[-+]?\.?[0-9]\d*|[-+]?[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?|\.[0-9]+)$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False

from json import load as json_load
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, type=str, required=True)
    parser.add_argument("--out", default="out.json", type=str, required=False)
    args = parser.parse_args()

    json_data = None
    with open(args.json, 'r') as jf:
        json_data = json_load(jf)

    print('Writing submission file to {}...'.format(args.out))
    data = {}
    for uuid in json_data:
    		if(is_number(json_data[uuid])):
    				data[uuid]=json_data[uuid]
    		else:
        		data[uuid] = []
        		data[uuid].append(json_data[uuid])

    with open(args.out, 'w', newline='', encoding='utf-8') as outfile:
        json.dump(data, outfile)
    
    
if __name__ == "__main__":
    main()
