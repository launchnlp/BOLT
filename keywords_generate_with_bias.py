from transformers import (
    GPT2TokenizerFast,
    AdamW,
    get_scheduler,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    AutoModelForCausalLM,
    BeamSearchScorer,
)
from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import sys
import time

from keywords_model_with_biases import GPTPromptTuningWithbiasesModelLM

prompt_file = "./keywords/prompts_15.txt"

keywords_dict = {
    "computer" : ["router", "Linux", "keyboard", "server"],
    "legal" : ["plea", "subpoena", "transcript", "bankrupt"],
    "military" : ["torpedo", "headquarters", "infantry", "battlefield"],
    "politics" : ["court", "culture", "communism", "capitalism"],
    "religion" : ["Bible", "church", "priest", "saint"],
    "science" : ["microscope", "mass", "mineral", "scientist"],
    "space" : ["meteor", "planet", "satellite", "astronaut"],
}

seq_len = int(sys.argv[1])
topic = sys.argv[2]
output_file = "./keywords/topic/" + topic + ".txt.len" + str(seq_len)

class Config:
    num_train_epochs = 50
    weight_decay = 0.01
    learning_rate = 0.4
    lr_scheduler_type = "linear"
    num_warmup_steps = 5
    max_train_steps = num_train_epochs
    
    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 10
    init_from_vocab = True
args = Config()

batch_size = 20

with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts_list = [line.strip() for line in f] 
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Initialize GPT2LM with soft prompt
    model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
        "gpt2-large",
        n_tokens=args.n_prompt_tokens,
        initialize_from_vocab=args.init_from_vocab,
        use_full_prompt=False,
    )
    model.cuda()

    for prompt in tqdm(prompts_list):
        keywords_word = [' '.join(keywords_dict[topic])] * batch_size
        prefixs = [prompt] * batch_size
        inputs = tokenizer(prefixs, return_tensors="pt")
        keywords = tokenizer([w for w in keywords_word], return_tensors="pt")['input_ids']
        inputs = inputs.to("cuda")
        keywords = keywords.to("cuda")
        model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1])
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "biases" in n],
                "weight_decay": args.weight_decay,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        model.eval()
        stored_sentence = [""] * batch_size
        success_idx = [-1] * batch_size
        start_time = time.time()
        for i in range(100):
            print("#################")
            loss, output_ids = model.soft_forward(**inputs, labels=inputs.input_ids, use_full_prompt=False, keywords=keywords)
            print(keywords_word)
            sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print(sentences)
            for idx in range(batch_size):
                if success_idx[idx] == -1:
                    if any([keyword in sentences[idx] for keyword in keywords_word[0].split(' ')]):
                        success_idx[idx] = i
                        stored_sentence[idx] = sentences[idx]
            if all([idx != -1 for idx in success_idx]):
                print("success")
                break
            loss.backward()
            if i % 1 == 0:
                optimizer.step()
                noise = [torch.normal(mean=0.01, std=0.01, size=model.biases[0].shape,
                                     device='cuda', requires_grad=False) for _ in range(len(model.biases))]
                for i in range(len(model.biases)):
                    model.biases[i].data = model.biases[i].data + noise[i]
            
        end_time = time.time()
        print("success_idx: ", success_idx)
        print("stored_sentence: ", stored_sentence)
        print("time: ", end_time - start_time)
        g.write('\n'.join(stored_sentence) + "\n\n")
        g.flush()
