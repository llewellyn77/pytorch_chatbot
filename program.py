import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import random
import os
import re

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the device to use for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Set the maximum input length
max_input_length = tokenizer.max_model_input_sizes['gpt2']

# Helper function to generate a response
def generate_response(prompt, temperature=0.96):
  # Encode the prompt and add the special tokens
  encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True, max_length=30, return_tensors='pt',truncation=False,attention_mask=1,pad_token_id=50256)
  encoded_prompt = encoded_prompt.to(device)

  # Generate a response
  with torch.no_grad():
    outputs = model.generate(input_ids=encoded_prompt, max_length=800, temperature=temperature)
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)

  # Clean up the response
  response = re.sub(r'\n', ' ', response)
  response = re.sub(r'  ', ' ', response)
  return response



# Chatbot loop
while True:
  user_input = input('You: ')
  if user_input.lower() in ['exit', 'quit', 'goodbye']:
    print('Bot: Goodbye!')
    break
  else:
    bot_response = generate_response(user_input)
    print('Bot:', bot_response)
