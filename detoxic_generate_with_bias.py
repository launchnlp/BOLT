# %%
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
import time
import sys

from model_with_biases import GPTPromptTuningWithbiasesModelLM


prompt_file = "./detoxic/sampled_1k_prompt.txt"
seq_len = int(sys.argv[1])
output_file = './detoxic/detoxic/gen_len' + str(seq_len) + '.txt'

class Config:
    num_train_epochs = 50
    weight_decay = 0.01
    learning_rate = 0.025
    lr_scheduler_type = "linear"
    num_warmup_steps = 5
    max_train_steps = num_train_epochs
    
    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 10
    init_from_vocab = True
args = Config()

batch_size = 15
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# Initialize GPT2LM with soft prompt
model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
    "gpt2-large",
    n_tokens=args.n_prompt_tokens,
    initialize_from_vocab=args.init_from_vocab,
    use_full_prompt=False,
)
model.cuda()
discriminator = AutoModelForSequenceClassification.from_pretrained("./checkpoints/replaced_vocab_roberta_for_jigsaw/")
discriminator.cuda()
model.init_discriminator(discriminator)


with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts = [line.strip() for line in f]

    for prompt in tqdm(prompts):
        prefixs = [prompt] * batch_size
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to("cuda")
        model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], 'non_toxic', 0.7)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "biases" in n or "trainable_weights" in n],
                "weight_decay": args.weight_decay,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        model.eval()
        minimun_loss = [100000] * batch_size
        stored_sentence = [""] * batch_size
        start_time = time.time()
        for i in range(8):
            if all([loss < 0.0003 for loss in minimun_loss]):
                break
            if i % 1 == 0:
                loss, output_ids, gpt_logit, senti_losses = model.soft_forward(**inputs, labels=inputs.input_ids, use_full_prompt=False)
                print("Decoding: ", loss)
                sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                print(sentences)
            loss.backward()
            if i % 1 == 0:
                optimizer.step()
                noise = [torch.normal(mean=0.01, std=0.01, size=model.biases[0].shape,
                                     device='cuda', requires_grad=False) for _ in range(len(model.biases))]
                for i in range(len(model.biases)):
                    model.biases[i].data = model.biases[i].data + noise[i]
            if i % 1 == 0:
                print(f"loss: {loss}")
                for idx in range(batch_size):
                    print(f"loss {idx}: senti loss: {senti_losses[idx]}")
                    if senti_losses[idx] < minimun_loss[idx]:
                        print(f"update minimun loss{idx}")
                        minimun_loss[idx] = senti_losses[idx]
                        stored_sentence[idx] = sentences[idx]
            
        end_time = time.time()
        print("minimun loss: ", minimun_loss)
        print("stored sentence: ", stored_sentence)
        print("time: ", end_time - start_time)
        g.write('\n'.join(stored_sentence) + "\n\n")
        g.flush()