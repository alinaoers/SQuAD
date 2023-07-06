import pickle
import numpy as np
# import torch

from dataset import DialogueDataset
from dataset.multiwoz import load_multiwoz
from dgac import dgac_one_stage, dgac_two_stage
from embedders import CachedEmbeddings
from dialogue_graph.frequency_dialogue_graph import FrequencyDialogueGraph



def proccess_new_slot(used_slots, slot_name, slot_value, running_dict):
  if (slot_name, slot_value) not in used_slots:
    running_dict['slots'][slot_name] = slot_value
    used_slots.append((slot_name, slot_value))


def prepare_data(graph, slot_types, with_dismatching_slots=False):
  res_list = []

  # for dialogue in graph.dialogues:
  #   for utterance in dialogue.utterances:
  for dialogue in graph:
    for utterance in dialogue:
      running_dict = {'utterance' : utterance.utterance, 'slots' : {}}
      if 'frames' in utterance.meta:
        for service in utterance.meta['frames']:
          if ('state' in service) and ('slot_values' in service['state']):
            for slot_name in service['state']['slot_values']:
              if slot_name in slot_types:
                slot_values = service['state']['slot_values'][slot_name]
                for slot_value in slot_values:

                  if (with_dismatching_slots == True):
                    running_dict['slots'][slot_name] = slot_value
                  elif (utterance.utterance.find(slot_value) != -1):
                    running_dict['slots'][slot_name] = slot_value
        
          if ('slots' in service) and ('slot' in service['slots']) and ('value' in service['slots']):
            slot_name = service['slots']['slot']
            slot_value = service['slots']['value']
            if (with_dismatching_slots != True) and (utterance.utterance.find(slot_value) != -1):
              running_dict['slots'][slot_name] = slot_value

      if (len(running_dict['slots']) != 0):  
        res_list.append(running_dict) 
     
  return res_list


def get_values_indices(values, length=5):
  sorted, indices = torch.sort(values, descending=True)
  if (length < sorted[sorted>=0].numpy().size):
    length = sorted[sorted>=0].numpy().size
  sorted = sorted[0, 0:length]
  indices = indices[0, 0:length]
  return sorted, indices


def get_indices_pairs(start_ind, end_ind, min_ind):
  start_ind = start_ind[start_ind>=min_ind]
  starts = []
  ends = []
  for start_val in start_ind:
    for end_val in end_ind:
      if end_val >= start_val and end_val - start_val < 6:
        starts.append(start_val)
        ends.append(end_val)
  return starts, ends


def get_answers_indices(outputs, inputs):
  start_logits_values, start_logits_indices = get_values_indices(outputs.start_logits)
  length = start_logits_values.numpy().size
  end_logits_values, end_logits_indices = get_values_indices(outputs.end_logits, length=length)
  min_ind = inputs.token_type_ids.argmax()
  # print(start_logits_values)
  starts, ends = get_indices_pairs(start_logits_indices, end_logits_indices, min_ind)
  # print(starts)
  return starts, ends


def get_num_right_pred_slots(data, questions, model, tokenizer):
  right_pred_slots = 0
  i = 0

  for running_dict in data:
    i += 1 
    true_slots_dict = running_dict['slots']
    pred_slots = []
    running_right_pred_slots = 0

    for question in questions:
      inputs = tokenizer(question, running_dict['utterance'], return_tensors="pt")
      with torch.no_grad():
        outputs = model(**inputs)
        starts, ends = get_answers_indices(outputs, inputs)
        if (len(starts) > 0 and len(ends) > 0):
          predict_slot = inputs.input_ids[0, starts[0] : ends[0] + 1]
          slot = tokenizer.decode(predict_slot)
          # print(slot)
          pred_slots.append(slot)
    pred_slots = np.unique(pred_slots)

    for slot in pred_slots:
      if slot in true_slots_dict.values():
        running_right_pred_slots += 1
        # print(running_right_pred_slots)
    right_pred_slots += running_right_pred_slots

    if (i % 5000) == 0 and i != 0:
      print('next 5k done')

  return right_pred_slots
  