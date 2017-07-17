"""
A simple averaging model.

In its default settings, this is the baseline unigram (Yu, 2014) approach
http://arxiv.org/abs/1412.1632 of training (M, b) such that:

    f(q, a) = sigmoid(q * M * a.T + b)

However, rather than a dot-product, the MLP comparison is used as it works
dramatically better.

This model can also represent the Deep Averaging Networks
(http://cs.umd.edu/~miyyer/pubs/2015_acl_dan.pdf) with this configuration:

    inp_e_dropout=0 inp_w_dropout=1/3 deep=2 "pact='relu'"

The model also supports preprojection of embeddings (not done by default;
wproj=True), though it doesn't do a lot of good it seems - the idea was to
allow mixin of NLP flags.


Performance:
    * anssel-yodaqa:
      valMRR=0.334864 (dot)
"""

from __future__ import print_function
from __future__ import division


from keras.layers import TimeDistributed, Dense
from keras.layers.merge import Average

import pysts.kerasts.blocks as B


def config(c):
    c['l2reg'] = 1e-5

    # word-level projection before averaging
    c['wproject'] = False
    c['wdim'] = 1
    c['wact'] = 'linear'

    c['deep'] = 0
    c['nnact'] = 'relu'
    c['nninit'] = 'glorot_uniform'

    c['project'] = True
    c['pdim'] = 1
    c['pact'] = 'tanh'

    # model-external:
    c['inp_e_dropout'] = 1/3
    c['inp_w_dropout'] = 0
    # anssel-specific:
    c['ptscorer'] = B.mlp_ptscorer
    c['mlpsum'] = 'sum'
    c['Ddim'] = 1


def prep_model(model, N, s0pad, s1pad, c):
    winputs = ['e0', 'e1']

    model = TimeDistributed(Average())(winputs)
    
    bow_last = ('e0b', 'e1b')

    return bow_last
