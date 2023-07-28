from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import torch

torch.manual_seed(0)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

context = tokenizer.encode("The future of Artificial Intelligence is", return_tensors="pt")#pt stands for pytorch

model = GPT2LMHeadModel.from_pretrained('gpt2')
      
#-----GREEDY SEARCH-----
greedy_output = model.generate(context, max_length=50, pad_token_id = 50256)

print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

print(100 * '-')

#-----BEAM SEARCH-----

beam_output = model.generate(
    context,
    max_new_tokens=40,
    num_beams=5,
    num_return_sequences=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
    pad_token_id = 50256,
)

for i, beam_output in enumerate(beam_output):
  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

print(100 * '-')

#-----TOP-K SEARCH-----

topk_output = model.generate(
    context,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    pad_token_id = 50256

)

print(tokenizer.decode(topk_output[0], skip_special_tokens=True))


