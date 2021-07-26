# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
#from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import ( 
    BertLMPredictionHead, 
    BertPreTrainedModel, 
    BertEmbeddings, 
    BertSelfAttention, 
    BertSelfOutput, 
    BertAttention, 
    BertIntermediate, 
    BertOutput, 
    BertEncoder, 
    BertPooler, 
    BertPredictionHeadTransform, 
    BertOnlyMLMHead,
    BertSelfAttention
)
from transformers import BertModel


logger = logging.get_logger(__name__)



class CVAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs




@dataclass
class CVBertOutput(ModelOutput):
    sequence_output: torch.FloatTensor = None
    user_group_scores: torch.FloatTensor = None
    post_mu: Optional[torch.FloatTensor] = None  #FIXME
    post_logvar: Optional[torch.FloatTensor] = None #FIXME
    prior_mu: torch.FloatTensor = None #FIXME
    prior_logvar: torch.FloatTensor = None #FIXME
    #past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    #hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #attentions: Optional[Tuple[torch.FloatTensor]] = None
    #cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
   


@dataclass
class CVBertForTrainingOutput(ModelOutput):
    loss: torch.FloatTensor = None
    masked_lm_loss: torch.FloatTensor = None
    elbo_loss: torch.FloatTensor = None
    prediction_logits: Optional[torch.FloatTensor] = None
    user_group_logits: Optional[torch.FloatTensor] = None



class CVBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.output = BertOutput(config)

        self.cv_attention = CVAttention(config)
     
        # posterior network takes input both x (hidden size = 768) and y (training signal about user group)
        self.posterior_net_fc1 = nn.Linear(config.hidden_size + config.y_input_dim, config.z_hidden_dim)
        # output dim is 2 * z_dim, because outputs both z_mean (size z_dim) and z_variance (also size z_dim)
        self.posterior_net_fc2 = nn.Linear(config.z_hidden_dim,  config.z_dim * 2)

        # prior network takes only x (hidden size = 768)
        self.prior_net_fc1 = nn.Linear(config.hidden_size, config.z_hidden_dim)
        self.prior_net_fc2 = nn.Linear(config.z_hidden_dim, config.z_dim * 2)
        self.z_dropout = nn.Dropout(config.z_dropout_prob)

        # In between we sample z: using z_mean and z_variance, we produce a random variable of size z_dim

        # y-predictor network takes as input x and the predicted z (this comes from the prior net!)
        # this is actually the second component of the prior network in the CVAE schema
        #input is x (hidden size = 768) + z (z_dim)
        self.y_predictor_net_fc = nn.Linear(config.hidden_size + config.z_dim, config.y_output_dim)


        # Takes in a concatenation of x (hidden size = 768) and z, and maps it back to hidden size (768)
        # FIXME: Is this layer necessary? Or even a good idea?
        # Meshes information in z and x together. Do we rather want to keep them separate?
        self.combination_fc = nn.Linear(config.hidden_size + config.z_dim, config.hidden_size)

        self.z_dim = config.z_dim


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        #output_attentions=False,
        #output_hidden_states=False,
        return_dict=False,
        user_group_labels=None,
    ):



        # IMPORTANT FIXME: Im removing the initial attention layer from CVBert layer for now
        # This attention is likely to start with garbage, maybe making learning more difficult
        # Can also be activated later in the training.

        '''
        #FIXME: What is head mask??? 
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cv_attention_outputs = self.cv_attention(
            hidden_states,
            attention_mask,
            head_mask,
            #output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        cv_x = cv_attention_outputs[0]
        '''
        cv_x = hidden_states

        # Sampling z
        post_mu = None
        post_logvar = None

        # If in training, run also the posterior network:
        if user_group_labels is not None:
            batch_size = cv_x.shape[0]
            sequence_len = cv_x.shape[1]
            user_group_labels_broadcasted = user_group_labels.view(-1, 1).repeat(1, sequence_len).view(batch_size, sequence_len, 1)
            
            post_inter = self.z_dropout(F.relu(self.posterior_net_fc1(torch.cat([cv_x, user_group_labels_broadcasted], dim=2))))
            post_mulogvar = self.posterior_net_fc2(post_inter)
            post_mu, post_logvar = post_mulogvar[:, :, :self.z_dim], post_mulogvar[:, :, self.z_dim:]


        # prior network is run whether in training or in test:
        prior_inter = self.z_dropout(F.relu(self.prior_net_fc1(cv_x)))
        prior_mulogvar = self.prior_net_fc2(prior_inter)
        prior_mu, prior_logvar = prior_mulogvar[:, :, :self.z_dim], prior_mulogvar[:, :, self.z_dim:]



        
        z_sample_prior = self.sample_gaussian(prior_mu, prior_logvar)
        
        if user_group_labels is not None: #during training
            z_sample_posterior = self.sample_gaussian(post_mu, post_logvar)
            y_prediction = F.log_softmax(self.y_predictor_net_fc(torch.cat([cv_x, z_sample_posterior], dim=2)), dim=2)
        else:    
            y_prediction = F.log_softmax(self.y_predictor_net_fc(torch.cat([cv_x, z_sample_prior], dim=2)), dim=2)

        # Combining z with hidden representations
        combined_representation = torch.cat([cv_x, z_sample_prior], dim=2)
        sequence_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, combined_representation
        )

        '''
        if output_attentions:
            self_attentions = attention_outputs[1]
        else:
            self_attentions = None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        '''

        if not return_dict:
            return tuple(
                v
                for v in [
                    sequence_output,
                    y_prediction,
                    post_mu,
                    post_logvar,
                    prior_mu,
                    prior_logvar#,
                    #self_attentions
                ]
                if v is not None
            )
        return CVBertOutput(
            sequence_output=sequence_output,
            #past_key_values=next_decoder_cache,
            #hidden_states=all_hidden_states,
            user_group_scores=y_prediction,
            post_mu=post_mu,
            post_logvar=post_logvar,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            #attentions=self_attentions,
            #cross_attentions=all_cross_attentions,
        )

        return outputs


    def feed_forward_chunk(self, combined_representation):
        #intermediate_output = self.intermediate(attention_output)
        #layer_output = self.output(intermediate_output, attention_output)
        layer_output = self.combination_fc(combined_representation)
        return layer_output


    def sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample





class CVBert(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        

        self.cv_layer = CVBertLayer(config)

        #self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights() #FIXME; I think I couldnt take the pretrained model initialization properly
        
        # Can I here reset to the pretrained model encoder and embeddings?
        pretrained_model = BertModel.from_pretrained('bert-base-uncased')
        pretrained_model.train() 
        self.embeddings = pretrained_model.embeddings
        self.encoder = pretrained_model.encoder
        

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        user_group_labels=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        #print(head_mask)
        #print(type(head_mask))
        #sys.exit(1)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            #user_group_labels=user_group_labels
        )

        cv_outputs = self.cv_layer(
            encoder_outputs.last_hidden_state,
            attention_mask=extended_attention_mask, #FIXME: extended_attention_mask nedir? Aynen kullanabilir miyim?
            head_mask=None, #FIXME: is the head mask important?
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            #past_key_values=past_key_values,
            #use_cache=use_cache,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            user_group_labels=user_group_labels
        )


        #sequence_output = cv_outputs[0]
        if not return_dict:
            #return (sequence_output, pooled_output) + encoder_outputs[1:]
            return cv_outputs

        return CVBertOutput(
            sequence_output=cv_outputs.sequence_output,
            user_group_scores=cv_outputs.user_group_scores,
            post_mu=cv_outputs.post_mu,
            post_logvar=cv_outputs.post_logvar,
            prior_mu=cv_outputs.prior_mu,
            prior_logvar=cv_outputs.prior_logvar,
            #past_key_values=encoder_outputs.past_key_values,
            #hidden_states=encoder_outputs.hidden_states,
            #attentions=encoder_outputs.attentions,     
        )


#Entry point
class CVBertForTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.cvbert = CVBert(config)
        self.cls = BertOnlyMLMHead(config)

        self.training_step = torch.tensor(0, requires_grad = False)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        user_group_labels=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        user_group_labels (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the user group prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.cvbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            user_group_labels=user_group_labels
        )

        if not return_dict:
            sequence_output = outputs[0]
            user_group_scores = outputs[1]
            post_mu = outputs[2]
            post_logvar = outputs[3]
            prior_mu = outputs[4]
            prior_logvar = outputs[5]
        else:
            sequence_output = outputs.sequence_output
            user_group_scores = outputs.user_group_scores
            post_mu = outputs.post_mu
            post_logvar = outputs.post_logvar
            prior_mu = outputs.prior_mu
            prior_logvar = outputs.prior_logvar


        # This one is for MLM training:
        prediction_scores = self.cls(sequence_output)

        total_loss = None
        if labels is not None and user_group_labels is not None:

            # FIXME: Nature of this loss function and user_group_scores/user_group_labels
            # needs further thought.

            # USER GROUP GOLD STANDARD:
            print(user_group_labels.tolist())

            #initial user_group_labels size is [batch_size, 1]
            # broadcast this to [batch_size, sequence_len], and then flatten this
            # FIXME: Maybe design so that we predict one user_group_label (y) for one tweet
            sequence_len = user_group_scores.shape[1]
            classification_dim = user_group_scores.shape[2]
            broadcasted_user_group_labels = user_group_labels.view(-1, 1).repeat(1, sequence_len).view(1, -1).squeeze()

            # USER GROUP PREDICTIONS:
            flattened_user_group_scores = user_group_scores.view(-1, classification_dim)
            _, predictions = torch.max(user_group_scores[:,0], dim=1) #FIXME: was 2 for token-level y calculation
            print(predictions.tolist())
            #FIXME: Is this flattening of the either terms correct?

            # I probably need a vectorized y representation and a suitable loss function here
            ylossfct = torch.nn.NLLLoss()
            y_loss = ylossfct(flattened_user_group_scores, broadcasted_user_group_labels)

            print('\n\n---- Step: %d -----\n' % self.training_step.item())
            print('y loss:\t\t\t\t%.4f\n' % y_loss.item())

            # Suggestion: Dont train with high LR for KLD loss at first, do "annealing"
            # FIXME: Does this make sense? Is 20000 enough of a scale?
            kl_weights = torch.minimum(self.training_step / 20000, torch.tensor(1.0))
            #print('kl weights:', kl_weights)
            
            KLD = self.gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
            print('KLD (z) loss:\t\t\t%.4f\n' % KLD.item())

            elbo_loss = y_loss + kl_weights * KLD
            print('elbo loss (y_loss + kl_weights * KLD): %.4f\n' % elbo_loss.item())

            mlm_loss_fct = CrossEntropyLoss()
            masked_lm_loss = mlm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            print('masked_lm_loss:\t\t\t%.4f\n' % masked_lm_loss.item())
             
            total_loss = masked_lm_loss + elbo_loss #FIXME: This 10 may be likely a very bad idea

            print('total loss:\t\t\t%.4f\n' % total_loss.item())

            self.training_step += 1


        if not return_dict:
            output = outputs
            return ((total_loss,masked_lm_loss,elbo_loss) + output) if total_loss is not None else output

        return CVBertForTrainingOutput(
            loss=total_loss,
            masked_lm_loss=masked_lm_loss,
            elbo_loss=elbo_loss,
            prediction_logits=prediction_scores,
            user_group_logits=user_group_scores
        )


    # BIG QUESTION MARK!
    # Need to check Kullback-Liebler formula for multivariate-Gaussian
    # And then write this function in a vectorized manner, instead of looping obver batch examples and tokens in input
    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        #kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
        #                       - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
        #                       - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)

        kld = torch.tensor([0.0]).to('cuda')
        for ex in range(recog_mu.shape[0]):
            for token in range(recog_mu.shape[1]):
                 mu1 = prior_mu[ex,token,:].to('cuda')
                 mu2 = recog_mu[ex,token,:].to('cuda')
                 sigma1 = prior_logvar[ex,token,:].to('cuda')
                 sigma2 = recog_logvar[ex,token,:].to('cuda')
                 kld += torch.tensor([-0.5]).to('cuda') * torch.sum(torch.tensor([1]).to('cuda') + (sigma2 - sigma1) 
                                 - torch.div(torch.pow(mu1 - mu2, 2), torch.exp(sigma1))
                                 - torch.div(torch.exp(sigma2), torch.exp(sigma1)), dim=0)
        return kld / (recog_mu.shape[0] * recog_mu.shape[1])
