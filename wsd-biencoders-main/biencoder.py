'''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
from sklearn.metrics import f1_score
import os
import sys
import time
import math
import copy
import argparse
from tqdm import tqdm
import pickle
from pytorch_transformers import *
import time
import random
import numpy as np
from tensorboardX import SummaryWriter
from wsd_models.util import *
from wsd_models.models import BiEncoderModel
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

parser = argparse.ArgumentParser(description='Gloss Informed Bi-encoder for WSD')

#training arguments
# --local_rank
#parser.add_argument('--local_rank', type=int)
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true',
	help='Flag to supress training progress bar for each epoch')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context-max-length', type=int, default=96)
parser.add_argument('--context-gloss-max-length', type=int, default=96)
parser.add_argument('--gloss-max-length', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--context-bsz', type=int, default=1)
parser.add_argument('--gloss-bsz', type=int, default=16)
parser.add_argument('--encoder-name', type=str, default='bert-base',
	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--ckpt', type=str,
	help='filepath at which to save best probing model (on dev set)', default='./wsd-biencoder')
parser.add_argument('--data-path', type=str,
	help='Location of top-level directory for the Unified WSD Framework', default='./WSD_Evaluation_Framework')

#sets which parts of the model to freeze ❄️ during training for ablation 
parser.add_argument('--freeze-context', action='store_true')
parser.add_argument('--freeze-gloss', action='store_true')
parser.add_argument('--tie-encoders', action='store_true')

#other training settings flags
parser.add_argument('--kshot', type=int, default=-1,
	help='if set to k (1+), will filter training data to only have up to k examples per sense')
parser.add_argument('--balanced', action='store_true',
	help='flag for whether or not to reweight sense losses to be balanced wrt the target word')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
	choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'all-test'],
	help='Which evaluation split on which to evaluate probe')

#uses these two gpus if training in multi-gpu
context_device = "cuda:0"
gloss_device = "cuda:0"
context_gloss_device = "cuda:1"

def tokenize_context_glosses(context_gloss_arr, tokenizer, max_len):
	glosses = []
	masks = []

	for gloss_text in context_gloss_arr:
		g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)+tokenizer.encode(gloss_text[:512])+tokenizer.encode(tokenizer.sep_token)]
		g_attn_mask = [1]*len(g_ids)
		g_fake_mask = [-1]*len(g_ids)
		g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
		g_ids = torch.cat(g_ids, dim=-1)
		g_attn_mask = torch.tensor(g_attn_mask)
		glosses.append(g_ids)
		masks.append(g_attn_mask)

	return glosses, masks

def tokenize_glosses(gloss_arr, tokenizer, max_len):
	glosses = []
	masks = []
	for gloss_text in gloss_arr:
		g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)+tokenizer.encode(gloss_text)+tokenizer.encode(tokenizer.sep_token)]
		g_attn_mask = [1]*len(g_ids)
		g_fake_mask = [-1]*len(g_ids)
		g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
		g_ids = torch.cat(g_ids, dim=-1)
		g_attn_mask = torch.tensor(g_attn_mask)
		glosses.append(g_ids)
		masks.append(g_attn_mask)

	return glosses, masks

#creates a sense label/ gloss dictionary for training/using the gloss encoder
#FSD
def load_and_preprocess_glosses(data, tokenizer, wn_senses, max_len=-1):
	sense_glosses = {}
	sense_weights = {}

	gloss_lengths = []

	for sent in data:
		for _, lemma, pos, _, label in sent:
			if label == -1:
				continue #ignore unlabeled words
			else:
				#print(lemma)
				key = generate_key(lemma, pos)
				if key not in sense_glosses:
					#get all sensekeys for the lemma/pos pair
					sensekey_arr = wn_senses[key]
					#get glosses for all candidate senses
					gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]

					#preprocess glosses into tensors
					gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
					gloss_ids = torch.cat(gloss_ids, dim=0)
					gloss_masks = torch.stack(gloss_masks, dim=0)
					sense_glosses[key] = (gloss_arr, gloss_ids, gloss_masks, sensekey_arr)

					#intialize weights for balancing senses
					sense_weights[key] = [0]*len(gloss_arr)
					w_idx = sensekey_arr.index(label)
					sense_weights[key][w_idx] += 1
				else:
					#update sense weight counts
					w_idx = sense_glosses[key][3].index(label)
					sense_weights[key][w_idx] += 1
				
				#make sure that gold label is retrieved synset
				assert label in sense_glosses[key][3]

	#normalize weights
	for key in sense_weights:
		total_w = sum(sense_weights[key])
		sense_weights[key] = torch.FloatTensor([total_w/x if x !=0 else 0 for x in sense_weights[key]])

	return sense_glosses, sense_weights

def preprocess_context(tokenizer, text_data, bsz=1, max_len=-1):
	if max_len == -1: assert bsz==1 #otherwise need max_length for padding

	context_ids = []
	sents = []
	context_attn_masks = []

	example_keys = []

	context_output_masks = []
	instances = []
	labels = []

	#tensorize data
	for sent in text_data:
		c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])] #cls token aka sos token, returns a list with index
		o_masks = [-1]
		sent_insts = []
		sent_keys = []
		sent_labels = []

		#For each word in sentence...
		for idx, (word, lemma, pos, inst, label) in enumerate(sent):
			#tensorize word for context ids
			word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
			c_ids.extend(word_ids)

			#if word is labeled with WSD sense...
			if inst != -1 :
				#add word to bert output mask to be labeled
				o_masks.extend([idx]*len(word_ids))
				#track example instance id
				sent_insts.append(inst)
				#track example instance keys to get glosses
				ex_key = generate_key(lemma, pos)
				sent_keys.append(ex_key)
				sent_labels.append(label)
			else:
				#mask out output of context encoder for WSD task (not labeled)
				o_masks.extend([-1]*len(word_ids))

			#break if we reach max len
			if max_len != -1 and len(c_ids) >= (max_len-1):
				break

		c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)])) #aka eos token
		c_attn_mask = [1]*len(c_ids)
		o_masks.append(-1)
		c_ids, c_attn_masks, o_masks = normalize_length(c_ids, c_attn_mask, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])

		y = torch.tensor([1]*len(sent_insts), dtype=torch.float)
		#not including examples sentences with no annotated sense data
		if len(sent_insts) > 0:
			sents.append(sent)
			context_ids.append(torch.cat(c_ids, dim=-1))
			context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
			context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
			example_keys.append(sent_keys)
			instances.append(sent_insts)
			labels.append(sent_labels)

	#package data
	data = list(zip(sents, context_ids, context_attn_masks, context_output_masks, example_keys, instances, labels))

	#batch data if bsz > 1
	if bsz >= 1:
		print('Batching data with bsz={}...'.format(bsz))
		batched_data = []
		for idx in range(0, len(data), bsz):
			if idx+bsz <=len(data): b = data[idx:idx+bsz]
			else: b = data[idx:]
			sents = [x for x,_,_,_,_,_,_ in b]
			context_ids = torch.cat([x for _,x,_,_,_,_,_ in b], dim=0)
			context_attn_mask = torch.cat([x for _, _,x,_,_,_,_ in b], dim=0)
			context_output_mask = torch.cat([x for _, _,_,x,_,_,_ in b], dim=0)
			example_keys = []
			for _, _,_,_,x,_,_ in b: example_keys.extend(x)
			instances = []
			for _, _,_,_,_,x,_ in b: instances.extend(x)
			labels = []
			for _, _,_,_,_,_,x in b: labels.extend(x)
			batched_data.append((sents, context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels))
		return batched_data
	else:  
		return data

def _train(tokenizer, train_data, model, gloss_dict, optim, schedule, criterion, gloss_bsz=-1, max_grad_norm=1.0, multigpu=False, silent=False, train_steps=-1):
	model.train()
	total_loss = 0.
	context_gloss = True
	start_time = time.time()

	train_data = enumerate(train_data)
	if not silent: train_data = tqdm(list(train_data))
	total_pred = 0
	correct_pred = 0
	total_pred_res = []
	total_label = []
	for i, (sent, context_ids, context_attn_mask, context_output_mask, example_keys, _, labels) in train_data:
		# print('sent:', len(sent), sent)
		# print('context_ids:', context_ids.shape)
		# print('example_keys:', len(example_keys), example_keys)
		# print('labels:', len(labels), labels)
		#print('example_keys:', example_keys, 'labels:', labels)
		#reset model
		model.zero_grad()
		#run example sentence(s) through context encoder
		if multigpu:
			context_ids = context_ids.to(context_device)
			context_attn_mask = context_attn_mask.to(context_device)
		else:
			context_ids = context_ids.cuda()
			context_attn_mask = context_attn_mask.cuda()
		# print('train fsd')
		context_output, example_len = model.context_forward(context_ids, context_attn_mask, context_output_mask)
		#print('context_output:', type(context_output))
		loss = 0.
		gloss_sz = 0
		context_sz = len(labels)
		#FSD
		# print('-----len----', len(context_output.split(1,dim=0)), len(sent) ) #54 2
		for j, (key, label) in enumerate(zip(example_keys, labels)):
			# print('--------context_output:', type(context_output),'--------')
			# print(context_output)
			output = context_output.split(1, dim=0)[j]

			for index, elen in enumerate(example_len):
				if j <= elen:
					sentence = sent[index]
					# print('j index', j, index)
					break

			#print('sentence', sentence)
			#print('output:', output.shape) #[1, 768]
			#run example's glosses through gloss encoder
			gloss_arr, gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
			# print('---------')
			# print('key', key, 'label', label)
			# print('gloss_ids', gloss_ids.shape)
			# print('gloss arr', gloss_arr)
			# print('----------')
			#get contenxt gloss
			context_gloss_arr = []
			for gloss_sent in gloss_arr:
				context_gloss = ''
				for word, lemma, pos, sense_inst, sense_label in sentence:
					#print('sent_word:', sent_word)
					#word, lemma, pos, sense_inst, sense_label
					if sense_inst == -1:
						context_gloss += word +' '
					else:
						context_gloss += gloss_sent +' '
					# if context_gloss[-1] == ' ':
					# 	context_gloss[-1] = ''
				context_gloss_arr.append(context_gloss)
			#print(context_gloss_arr)

			context_gloss_ids, context_gloss_attn_mask = tokenize_context_glosses(context_gloss_arr, tokenizer, args.context_gloss_max_length)
			context_gloss_ids = torch.cat(context_gloss_ids, dim=0)
			context_gloss_attn_mask = torch.stack(context_gloss_attn_mask, dim=0)

			#print('gloss_arr', gloss_arr)
			#print('sense_keys', sense_keys)

			if multigpu:
				gloss_ids = gloss_ids.to(gloss_device)
				gloss_attn_mask = gloss_attn_mask.to(gloss_device)

				context_gloss_ids = context_gloss_ids.to(context_gloss_device)
				context_gloss_attn_mask = context_gloss_attn_mask.to(context_gloss_device)
			else:
				gloss_ids = gloss_ids.cuda()
				gloss_attn_mask = gloss_attn_mask.cuda()

				context_gloss_ids = context_gloss_ids.cuda()
				context_gloss_attn_mask = context_gloss_attn_mask.cuda()

			gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
			gloss_output = gloss_output.transpose(0,1) #[768, number of definitions]

			#print('context_gloss_ids:', context_gloss_ids.shape)

			context_gloss_output = model.context_gloss_forward(context_gloss_ids, context_gloss_attn_mask)
			#exit()
			context_gloss_output = context_gloss_output.transpose(0,1) #[768, number of definitions]
			#print('context_gloss_output', context_gloss_output.shape)

			#print('key:', key, 'label:', label, 'gloss_output:', gloss_output.shape)
			#get cosine sim of example from context encoder with gloss embeddings
			if multigpu:
				output = output.cpu()
				gloss_output = gloss_output.cpu()
				context_gloss_output = context_gloss_output.cpu()

			if context_gloss:
				output = torch.mm(output, gloss_output + context_gloss_output)
			else:
				output = torch.mm(output, gloss_output)
			# print('------mm outpu------', output.shape)

			#exit()

			#get label and calculate loss
			idx = sense_keys.index(label)
			label_tensor = torch.tensor([idx])
			if not multigpu: label_tensor = label_tensor.cuda()

			#looks up correct candidate senses criterion
			#needed if balancing classes within the candidate senses of a target word
			#print('*** output', output.shape, '****label_tensor', label_tensor.shape)
			#print(output, label_tensor)
			#print('equal: ',torch.argmax(output)==label_tensor[0])
			total_label.append(idx)

			total_pred_res.append(np.argmax(output.detach().cpu().numpy()))
			total_pred += 1
			if torch.argmax(output)==label_tensor[0]:
				correct_pred += 1

			#exit()
			loss += criterion[key](output, label_tensor)
			gloss_sz += gloss_output.size(-1)

			if gloss_bsz != -1 and gloss_sz >= gloss_bsz:
				#update model
				total_loss += float(loss.item())
				loss=loss/gloss_sz
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
				optim.step()
				schedule.step() # Update learning rate schedule

				#reset loss and gloss_sz
				loss = 0.
				gloss_sz = 0

				#reset model
				model.zero_grad()

				#rerun context through model
				context_output, _ = model.context_forward(context_ids, context_attn_mask, context_output_mask)

		#update model after finishing context batch
		if gloss_bsz != -1: loss_sz = gloss_sz
		else: loss_sz = context_sz
		if loss_sz > 0:
			total_loss += float(loss.item())
			loss=loss/loss_sz
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optim.step()
			schedule.step() # Update learning rate schedule

		#stop epoch early if number of training steps is reached
		if train_steps > 0 and i+1 == train_steps: break
	print('------train loss--------: ', total_loss)
	print('-------train acc---------:', correct_pred/total_pred)
	# print(type(total_label), type(total_pred_res))
	# #print('-------train f1---------:', f1_score(total_label, total_pred_res, zero_division=1))
	# print("F1-Score:{:.4f}".format(f1_score(total_label, total_pred_res)))
	return model, optim, schedule, total_loss, correct_pred/total_pred
#eval_preds = _eval(tokenizer, criterion, -1, semeval2007_data, model, semeval2007_gloss_dict, multigpu=args.multigpu)
def _eval(tokenizer, criterion, gloss_bsz, eval_data, model, gloss_dict, multigpu=False):
	context_gloss = True
	model.eval()
	eval_preds = []
	total_loss = .0
	total_pred = 0
	correct_pred = 0
	for sent, context_ids, context_attn_mask, context_output_mask, example_keys, insts, labels in eval_data:
		# print('sent:', len(sent))
		# print('context_ids:', context_ids.shape)
		# print('example_keys:', len(example_keys))
		#print('labels:', len(labels))
		loss = 0.
		gloss_sz = 0
		context_sz = len(labels)
		with torch.no_grad(): 
			#run example through model
			if multigpu:
				context_ids = context_ids.to(context_device)
				context_attn_mask = context_attn_mask.to(context_device)
			else:
				context_ids = context_ids.cuda()
				context_attn_mask = context_attn_mask.cuda()
			context_output, example_len = model.context_forward(context_ids, context_attn_mask, context_output_mask)
			# print('context_forward:', context_output.shape)

			for j, (output, key, inst, label) in enumerate(zip(context_output.split(1,dim=0), example_keys, insts, labels)):
				#run example's glosses through gloss encoder
				for index, elen in enumerate(example_len):
					if j <= elen:
						sentence = sent[index]
						# print('j index', j, index)
						break

				gloss_arr, gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]

				# get contenxt gloss
				context_gloss_arr = []
				for gloss_sent in gloss_arr:
					context_gloss = ''
					for word, lemma, pos, sense_inst, sense_label in sentence:
						# print('sent_word:', sent_word)
						# word, lemma, pos, sense_inst, sense_label
						if sense_inst == -1:
							context_gloss += word + ' '
						else:
							context_gloss += gloss_sent + ' '
					# if context_gloss[-1] == ' ':
					# 	context_gloss[-1] = ''
					context_gloss_arr.append(context_gloss)
				context_gloss_ids, context_gloss_attn_mask = tokenize_context_glosses(context_gloss_arr, tokenizer, args.context_gloss_max_length)
				context_gloss_ids = torch.cat(context_gloss_ids, dim=0)
				context_gloss_attn_mask = torch.stack(context_gloss_attn_mask, dim=0)

				if multigpu:
					gloss_ids = gloss_ids.to(gloss_device)
					gloss_attn_mask = gloss_attn_mask.to(gloss_device)
					context_gloss_ids = context_gloss_ids.to(context_gloss_device)
					context_gloss_attn_mask = context_gloss_attn_mask.to(context_gloss_device)
				else:
					gloss_ids = gloss_ids.cuda()
					gloss_attn_mask = gloss_attn_mask.cuda()
					context_gloss_ids = context_gloss_ids.cuda()
					context_gloss_attn_mask = context_gloss_attn_mask.cuda()

				gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
				gloss_output = gloss_output.transpose(0,1)
				context_gloss_output = model.context_gloss_forward(context_gloss_ids, context_gloss_attn_mask)
				context_gloss_output = context_gloss_output.transpose(0, 1)  # [768, number of definitions]

				#get cosine sim of example from context encoder with gloss embeddings
				if multigpu:
					output = output.cpu()
					gloss_output = gloss_output.cpu()
					context_gloss_output = context_gloss_output.cpu()

				if context_gloss:
					output = torch.mm(output, gloss_output + context_gloss_output)
				else:
					output = torch.mm(output, gloss_output)

				pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
				pred_label = sense_keys[pred_idx]
				eval_preds.append((inst, pred_label))

				#eval loss
				idx = sense_keys.index(label)
				label_tensor = torch.tensor([idx])
				total_pred += 1
				if torch.argmax(output) == label_tensor[0]:
					correct_pred += 1

				# exit()
				if key not in criterion.keys():
					criterion[key] = torch.nn.CrossEntropyLoss(reduction='none')

				loss += criterion[key](output, label_tensor)
				if not multigpu: label_tensor = label_tensor.cuda()

				loss += criterion[key](output, label_tensor)
				gloss_sz += gloss_output.size(-1)


				if gloss_bsz != -1 and gloss_sz >= gloss_bsz:
					# update model
					total_loss += float(loss.item())
					loss = loss / gloss_sz
					# reset loss and gloss_sz
					loss = 0.
					gloss_sz = 0


			# update model after finishing context batch
			if gloss_bsz != -1:
				loss_sz = gloss_sz
			else:
				loss_sz = context_sz
			if loss_sz > 0:
				total_loss += float(loss.item())
				loss = loss / loss_sz
	print('------eval loss--------: ', total_loss)
	print('------eval acc--------: ', correct_pred/total_pred)
	return eval_preds, total_loss, correct_pred/total_pred

def train_model(args):
	target_sense = 'hospital'

	print('Training WSD bi-encoder model...')
	if args.freeze_gloss: assert args.gloss_bsz == -1 #no gloss bsz if not training gloss encoder, memory concerns

	#create passed in ckpt dir if doesn't exist
	if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

	'''
	LOAD PRETRAINED TOKENIZER, TRAIN AND DEV DATA
	'''
	print('Loading data + preprocessing...')
	sys.stdout.flush()

	tokenizer = load_tokenizer(args.encoder_name)

	#loading WSD (semcor) data
	train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')
	#train_data = load_data_signal_sens(train_path, 'semcor', target_sense)
	#print('train_data: ', train_data[0])
	#filter train data for k-shot learning
	if args.kshot > 0: train_data = filter_k_examples(train_data, args.kshot)

	#dev set = semeval2007
	semeval2007_path = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/')
	semeval2007_data = load_data(semeval2007_path, 'semeval2007')
	#semeval2007_data = load_data_signal_sens(train_path, 'semcor', target_sense)

	#load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)
	wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
	wn_senses = load_wn_senses(wn_path)
	train_gloss_dict, train_gloss_weights = load_and_preprocess_glosses(train_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
	semeval2007_gloss_dict, _ = load_and_preprocess_glosses(semeval2007_data, tokenizer, wn_senses, max_len=args.gloss_max_length)
	#print('train_gloss_dict:', train_gloss_dict)
	#exit()
	#preprocess and batch data (context + glosses)
	train_data = preprocess_context(tokenizer, train_data, bsz=args.context_bsz, max_len=args.context_max_length)
	semeval2007_data = preprocess_context(tokenizer, semeval2007_data, bsz=1, max_len=args.context_max_length)

	epochs = args.epochs
	overflow_steps = -1
	t_total = len(train_data)*epochs

	#if few-shot training, override epochs to calculate num. epochs + steps for equal training signal
	if args.kshot > 0:
		#hard-coded num. of steps of fair kshot evaluation against full model on default numer of epochs
		NUM_STEPS = 181500 #num batches in full train data (9075) * 20 epochs 
		num_batches = len(train_data)
		epochs = NUM_STEPS//num_batches #recalculate number of epochs
		overflow_steps = NUM_STEPS%num_batches #num steps in last overflow epoch (if there is one, otherwise 0)
		t_total = NUM_STEPS #manually set number of steps for lr schedule
		if overflow_steps > 0: epochs+=1 #add extra epoch for overflow steps
		print('Overriding args.epochs and training for {} epochs...'.format(epochs))

	''' 
	SET UP FINETUNING MODEL, OPTIMIZER, AND LR SCHEDULE
	'''
	model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context, tie_encoders=args.tie_encoders)

	#speeding up training by putting two encoders on seperate gpus (instead of data parallel)
	if args.multigpu: 
		model.gloss_encoder = model.gloss_encoder.to(gloss_device)
		model.context_encoder = model.context_encoder.to(context_device)
		model.context_gloss_encoder = model.context_gloss_encoder.to(context_gloss_device)
	else:
		model = model.cuda()
		model = torch.nn.parallel.DistributedDataParallel(model)

	criterion = {}
	if args.balanced:
		for key in train_gloss_dict:
			criterion[key] = torch.nn.CrossEntropyLoss(reduction='none', weight=train_gloss_weights[key])
	else:
		for key in train_gloss_dict:
			criterion[key] = torch.nn.CrossEntropyLoss(reduction='none')

	#optimize + scheduler from pytorch_transformers package
	#this taken from pytorch_transformers finetuning code
	weight_decay = 0.0 #this could be a parameter
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	adam_epsilon = 1e-8
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon)
	schedule = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup, t_total=t_total)

	'''
	TRAIN MODEL ON SEMCOR DATA
	'''

	#best_dev_f1 = 0.
	print('Training probe...')
	sys.stdout.flush()
	writer = SummaryWriter('runs/ex-context-gloss')
	best_eval_loss = 999.
	for epoch in range(1, epochs+1):
		#if last epoch, pass in overflow steps to stop epoch early
		train_steps = -1
		if epoch == epochs and overflow_steps > 0: train_steps = overflow_steps

		#train model for one epoch or given number of training steps
		model, optimizer, schedule, train_loss, train_acc = _train(tokenizer, train_data, model, train_gloss_dict, optimizer, schedule, criterion, gloss_bsz=args.gloss_bsz, max_grad_norm=args.grad_norm, silent=args.silent, multigpu=args.multigpu, train_steps=train_steps)
		for _ in range(5):
			torch.cuda.empty_cache()
			# time.sleep(1)
		#eval model on dev set (semeval2007)
		eval_preds, eval_loss, eval_acc = _eval(tokenizer, criterion, args.gloss_bsz, semeval2007_data, model, semeval2007_gloss_dict, multigpu=args.multigpu)

		# #generate predictions file
		# pred_filepath = os.path.join(args.ckpt, 'tmp_predictions.txt')
		# with open(pred_filepath, 'w') as f:
		# 	for inst, prediction in eval_preds:
		# 		f.write('{} {}\n'.format(inst, prediction))
		#
		# #run predictions through scorer
		# gold_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt')
		# scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
		# _, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
		# print('Dev f1 after {} epochs = {}'.format(epoch, dev_f1))
		# sys.stdout.flush()
		#
		# if dev_f1 >= best_dev_f1:
		# 	print('updating best model at epoch {}...'.format(epoch))
		# 	sys.stdout.flush()
		# 	best_dev_f1 = dev_f1
		# 	#save to file if best probe so far on dev set
		# 	model_fname = os.path.join(args.ckpt, 'best_model.ckpt')
		# 	with open(model_fname, 'wb') as f:
		# 		torch.save(model.state_dict(), f)
		# 	sys.stdout.flush()

		#shuffle train set ordering after every epoch
		random.shuffle(train_data)
		writer.add_scalar('train_loss', train_loss, epoch)
		writer.add_scalar('train_acc', train_acc, epoch)
		writer.add_scalar('eval_loss', eval_loss, epoch)
		writer.add_scalar('eval_acc', eval_acc, epoch)
		if eval_loss < best_eval_loss:
			best_eval_loss = eval_loss
			model_fname = os.path.join(args.ckpt, 'best_model.ckpt')
			with open(model_fname, 'wb') as f:
				torch.save(model.state_dict(), f)
	return

def evaluate_model(args):
	print('Evaluating WSD model on {}...'.format(args.split))

	'''
	LOAD TRAINED MODEL
	'''
	model = BiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context)
	model_path = os.path.join(args.ckpt, 'best_model.ckpt')
	model.load_state_dict(torch.load(model_path))
	model = model.cuda()
	

	'''
	LOAD TOKENIZER
	'''
	tokenizer = load_tokenizer(args.encoder_name)

	'''
	LOAD EVAL SET
	'''
	eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(args.split))
	eval_data = load_data(eval_path, args.split)

	#load gloss dictionary (all senses from wordnet for each lemma/pos pair that occur in data)
	wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
	wn_senses = load_wn_senses(wn_path)
	gloss_dict, _ = load_and_preprocess_glosses(eval_data, tokenizer, wn_senses, max_len=32)

	eval_data = preprocess_context(tokenizer, eval_data, bsz=1, max_len=-1)

	'''
	EVALUATE MODEL
	'''
	eval_preds = _eval(eval_data, model, gloss_dict, multigpu=False)

	#generate predictions file
	pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(args.split))
	with open(pred_filepath, 'w') as f:
		for inst, prediction in eval_preds:
			f.write('{} {}\n'.format(inst, prediction))

	#run predictions through scorer
	gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
	scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
	p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
	print('f1 of BERT probe on {} test set = {}'.format(args.split, f1))

	return

if __name__ == "__main__":
	if not torch.cuda.is_available():
		print("Need available GPU(s) to run this model...")
		quit()

	#parse args
	args = parser.parse_args()
	#torch.cuda.set_device(args.local_rank)  # before your code runs
	#torch.distributed.init_process_group(backend='nccl')
	print(args)

	#set random seeds
	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)   
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic=True

	#evaluate model saved at checkpoint or...
	if args.eval: evaluate_model(args)
	#train model
	else: train_model(args)

#EOF