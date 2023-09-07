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
from transformers import AutoModelForSequenceClassification, GPT2ForSequenceClassification
import torch
from tqdm import tqdm
import time
import sys

from model_with_biases import GPTPromptTuningWithbiasesModelLM

prompt_file = "./sentiment/prompts_15.txt"
seq_len = int(sys.argv[1])
sentiment = sys.argv[2] # pos or neg
output_file = "./sentiment/sentiment/" + sys.argv[2] + ".txt.len" + str(seq_len)

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

batch_size = 20

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

model = GPTPromptTuningWithbiasesModelLM.from_pretrained(
    "gpt2-large",
    n_tokens=args.n_prompt_tokens,
    initialize_from_vocab=args.init_from_vocab,
    use_full_prompt=False,
)
model.cuda()
discriminator = AutoModelForSequenceClassification.from_pretrained("./checkpoints/replaced_vocab_roberta_for_yelp_polarity")
discriminator.cuda()
model.init_discriminator(discriminator)


with open(prompt_file, "r") as f, open(output_file, "w") as g:
    prompts = [line.strip() for line in f]

    for prompt in tqdm(prompts):
        prefixs = [prompt] * batch_size
        inputs = tokenizer(prefixs, return_tensors="pt")
        inputs = inputs.to("cuda")
        model.set_biases(batch_size, seq_len + inputs.input_ids.shape[1], sentiment)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "biases" in n or "trainable_weights" in n],
                "weight_decay": args.weight_decay,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        model.eval()
        minimum_loss = [100000] * batch_size
        stored_sentence = [""] * batch_size
        start_time = time.time()
        for i in range(8):
            if i % 1 == 0:
                loss, output_ids, gpt_logit, senti_losses = model.soft_forward(**inputs, labels=inputs.input_ids, use_full_prompt=False)
                print("Decoding: ", loss)
                sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                print(sentences)
                print(time.time()-start_time)

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
                    if senti_losses[idx] < minimum_loss[idx]:
                        print(f"update minimum loss{idx}")
                        minimum_loss[idx] = senti_losses[idx]
                        stored_sentence[idx] = sentences[idx]
        
        end_time = time.time()
        print("minimum loss: ", minimum_loss)
        print("stored sentence: ", stored_sentence)
        print("time: ", end_time - start_time)
        g.write('\n'.join(stored_sentence) + "\n\n")
        g.flush()
