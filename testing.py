from llm import tokenizer
from steg import encode, decode
from Helpers import detect, get_limit
import torch

token_match = 0

def mismatch(enc_ids, dec_ids):
  enc = [tokenizer.decode(t) for t in enc_ids]
  dec = [tokenizer.decode(t) for t in dec_ids]
  print(enc)
  print(dec)

  # compare how many indices are the same
  same = 0
  for i in range(len(dec_ids)):
    if enc_ids[i] == dec_ids[i]:
      same += 1
  print('num same: ', same)
  print('dec length: ', len(dec_ids))

def test(keys, h, m, delta, c):
  global token_match
  ct, tokens = encode(keys, h, m, delta, c)
  print('-------------------')
  recovered_counters, decode_tokens = decode(keys, h, ct, None, c)
  same = torch.equal(tokens['input_ids'], decode_tokens['input_ids'])
  print('Tokens match: ', same)
  if (same):
    token_match += 1
  if (not same):
    mismatch(tokens['input_ids'][0], decode_tokens['input_ids'][0])

  print('-------------------')
  print(ct)
  print(recovered_counters)

  for i in range(len(recovered_counters)):
    print(detect(get_limit(None), recovered_counters[i]))

m = [1]
l = len(m)
keys = [b'\0' * 32]
h = ['Hey! Want to get coffee? ', 'Would Friday work for you?']
c = 5

# deltas = [0.0001, .001, .01, .1, .5]
# search for what delta works best
# for delta in deltas:
#   print('******************')
#   print('Delta: ', delta)
#   test(keys, h, m, delta, c)

delta = 0.01
# test how many times encode and decode perfectly match
# for _ in range(15):
#   test(keys, h, m, delta, c)

test(keys, h, m, delta, c)

print('Token match: ', token_match)