from steg import encode, decode
from Helpers import detect, get_limit


# m = [1, 0, 1]
m = [1, 0]
# m = [1]
l = len(m)
# keys = [b'\0' * 32, b'\1' * 32, b'\2' * 32]
keys = [b'\0' * 32, b'\1' * 32]
# keys = [b'\0' * 32]
h = ['Hey! Want to get coffee?', 'Would Friday work for you?']
c = 5

# keys = [get_random_bytes(32), get_random_bytes(32), get_random_bytes(32)]
delta = 0.0001
ct, encode_tokens = encode(keys, h, m, delta, c)
print('-------------------')
recovered_counters, decode_tokens = decode(keys, h, ct, None, c)

print(ct)
print(recovered_counters)

print(detect(get_limit(None), recovered_counters[0]))

print(len(encode_tokens))
print(encode_tokens)
print(len(decode_tokens))
print(decode_tokens)