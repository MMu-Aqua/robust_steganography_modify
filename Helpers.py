import numpy as np
import math
from collections import Counter

def get_limit(num_watermarks):
  #! temporary constant
  return 1000

def get_keys_to_use(m, keys):
  return [key for key, flag in zip(keys, m) if flag == 1]

def sample_key(keys):
  random_index = np.random.randint(len(keys))
  return keys[random_index]

# T is the number of tokens generated
# s_g is the number of green list tokens
def detect(T, s_g):
  print (T, s_g)
  z = (2 * (s_g - T / 2)) / math.sqrt(T)
  return z > 2 #! temporary threshold

def generate_n_grams_with_counts(lst, n):
    """Generate n-grams from a list along with their counts."""
    n_grams = [tuple(lst[i:i + n]) for i in range(len(lst) - n + 1)]
    return Counter(n_grams)

def count_maintained_n_grams_with_frequency(list_1, list_2, n):
    """Count how many times n-grams from list 1 are maintained in list 2, considering their frequencies."""
    n_grams_list_1 = generate_n_grams_with_counts(list_1, n)
    n_grams_list_2 = generate_n_grams_with_counts(list_2, n)
    
    total_maintained = 0
    for n_gram, count in n_grams_list_1.items():
        total_maintained += min(count, n_grams_list_2.get(n_gram, 0))
    
    return total_maintained