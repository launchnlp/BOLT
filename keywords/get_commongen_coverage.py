import json
import os
import sys
input_file = sys.argv[1]

with open(input_file) as f:
    keywords_count = 0
    matched_keywords_count = 0
    for line in f.read().split('\n'):
        if line == '':
            continue
        line = eval(line)
        if "COLD" in input_file:
            keywords = line['constraints'].split(' ')
        elif "mucola" in input_file:
            keywords = eval(line['keywords'])['concept_set'].split('#')
        else:
            keywords = line['keywords'].split(' ')
        for keyword in keywords:
            keywords_count += 1
            if "COLD" in input_file:
                if keyword in line['generation_complete'][0]:
                    matched_keywords_count += 1
            elif "mucola" in input_file:
                if keyword in line['generations'][0]['text']:
                    matched_keywords_count += 1
            else:
                if keyword in line['generation']:
                    matched_keywords_count += 1
    print(matched_keywords_count / keywords_count)