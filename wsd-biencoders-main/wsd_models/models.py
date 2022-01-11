'''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
from torch.nn import functional as F
import math
import os
import sys
from pytorch_transformers import *

from wsd_models.util import *

def load_projection(path):
    proj_path = os.path.join(path, 'best_probe.ckpt')
    with open(proj_path, 'rb') as f: proj_layer = torch.load(f)
    return proj_layer

class PretrainedClassifier(torch.nn.Module):
    def __init__(self, num_labels, encoder_name, proj_ckpt_path):
        super(PretrainedClassifier, self).__init__()

        self.encoder, self.encoder_hdim = load_pretrained_model(encoder_name)

        if proj_ckpt_path and len(proj_ckpt_path) > 0:
            self.proj_layer = load_projection(proj_ckpt_path)
            #assert to make sure correct dims
            assert self.proj_layer.in_features == self.encoder_hdim
            assert self.proj_layer.out_features == num_labels
        else:
            self.proj_layer = torch.nn.Linear(self.encoder_hdim, num_labels)

    def forward(self, input_ids, input_mask, example_mask):
        output = self.encoder(input_ids, attention_mask=input_mask)[0]

        example_arr = []        
        for i in range(output.size(0)): 
            example_arr.append(process_encoder_outputs(output[i], example_mask[i], as_tensor=True))
        output = torch.cat(example_arr, dim=0)
        output = self.proj_layer(output)
        return output

class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

    def forward(self, input_ids, attn_mask):
        #encode gloss text
        # print('gloss input', input_ids.shape)
        if self.is_frozen:
            with torch.no_grad(): 
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        #training model to put all sense information on CLS token
        # print('gloss_output', gloss_output.shape)
        gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        # print('gloss_output', gloss_output.shape)
        return gloss_output

class ContextGlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context_gloss, tied_encoder=None):
        super(ContextGlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.context_gloss_encoder = tied_encoder
            _, self.context_gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.context_gloss_encoder, self.context_gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context_gloss

    def forward(self, input_ids, attn_mask):
        #encode gloss text
        #print('gloss input', input_ids.shape)
        if self.is_frozen:
            with torch.no_grad():
                context_gloss_output = self.context_gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_gloss_output = self.context_gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        #training model to put all sense information on CLS token
        #print('gloss_output', gloss_output.shape)
        context_gloss_output = context_gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        #print('gloss_output', gloss_output.shape)
        return context_gloss_output

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context

    def forward(self, input_ids, attn_mask, output_mask):
        #encode context
        if self.is_frozen:
            with torch.no_grad(): 
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        #print('input_ids: ', input_ids.shape, 'attn_mask: ', attn_mask.shape) #[4, 128]
        #average representations over target word(s)
        #print('context_output:', context_output.shape, context_output) #[4, 128, 768]
        example_arr = []        
        for i in range(context_output.size(0)): 
            example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        #print('example_arr:', len(example_arr)) #[7, 768]
        example_len = []
        total_example_len = 0
        for i in example_arr:
            #print(i.shape)
            total_example_len += i.size(0) - 1
            example_len.append(total_example_len)

        context_output = torch.cat(example_arr, dim=0)
        #print('context_output2:', context_output.shape) #[34, 768]


        return context_output, example_len

class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, freeze_context_gloss=False, tie_encoders=False):
        super(BiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
            self.context_gloss_encoder = ContextGlossEncoder(encoder_name, freeze_context_gloss, tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
            self.context_gloss_encoder = ContextGlossEncoder(encoder_name, freeze_context_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim
        assert self.context_encoder.context_hdim == self.context_gloss_encoder.context_gloss_hdim

    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)

    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)

    def context_gloss_forward(self, context_gloss_input, context_gloss_attn_mask):
        return self.context_gloss_encoder.forward(context_gloss_input, context_gloss_attn_mask)

#EOF