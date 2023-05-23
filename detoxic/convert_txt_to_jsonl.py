import json

# input_file = "gen_len20.txt"
# output_file = "gen_len20.jsonl"
# input_file = "/home/liuxincs/COLD_decoding/data/detoxic/extracted.txt"
# output_file = "/home/liuxincs/COLD_decoding/data/detoxic/extracted.jsonl"
input_file = "/data/xin/prompts_from_PPLM/toxicity_gpt2_large_gen_len20.txt"
output_file = "/data/xin/prompts_from_PPLM/toxicity_gpt2_large_gen_len20.jsonl"
prompt_file = "/data/xin/real-toxicity-prompts/sampled_1k_prompt.txt"

with open(input_file) as f, open(output_file, 'w') as o, open(prompt_file) as p:
    generations = [l for l in f.read().split('\n\n') if l != '']
    prompts = [l for l in p.read().split('\n') if l != '']
    assert len(generations) == len(prompts)
    for i in range(len(prompts)):
        line_dict = {'prompt':{}, "generations":[]}
        line_dict['prompt']['text'] = prompts[i]
        for g in generations[i].split('\n'):
            if g != '':
                line_dict['generations'].append({'text': g[len(prompts[i]):]})
        o.write(json.dumps(line_dict) + '\n')