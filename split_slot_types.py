import numpy as np
import random


def random_split_slot_types(all_types):
    num_train = 7
    num_dev = 3
    num_test = 8
    dev_types, test_types = [], []
    types_copy = all_types.copy()
    for i in range(num_dev):
        dev_types.append(random.choice(types_copy))
        types_copy.remove(dev_types[-1])
    for i in range(num_test):
        test_types.append(random.choice(types_copy))
        types_copy.remove(test_types[-1])
    train_types = types_copy.copy()
    return train_types, dev_types, test_types

def get_splitted_slot_types():
  domains = {'attraction': ['area', 'name', 'type'], 
           'bus': ['departure', 'destination'], 'hospital': ['department'], 
            'hotel': ['area', 'bookday', 'bookpeople',
            'bookstay', 'internet', 'name', 'parking',
            'pricerange', 'stars', 'type'], 
            'restaurant': ['bookday', 'bookpeople',
            'booktime', 'food', 'name', 'pricerange'], 
            'taxi': ['arriveby', 'departure',
            'destination', 'leaveat'], 
            'train': ['arriveby',
            'bookpeople', 'day', 'departure',
            'destination', 'leaveat']}
  all_types = np.unique(np.hstack(domains.values())).tolist()
  train_types, dev_types, test_types = random_split_slot_types(all_types)
  train_slot_types, dev_slot_types, test_slot_types = [], [], []
  for domain, domain_types in domains.items():
    for slot_type in domain_types:
      if slot_type in train_types:
        train_slot_types.append(f'{domain}-{slot_type}')
      if slot_type in dev_types:
        dev_slot_types.append(f'{domain}-{slot_type}')
      if slot_type in test_types:
        test_slot_types.append(f'{domain}-{slot_type}')
  return train_slot_types, dev_slot_types, test_slot_types