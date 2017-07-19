"""
KeraSTS interface for datasets of the Semantic Text Similarity task.
See data/sts/... for details and actual datasets.

Training example:
    tools/train.py cnn sts data/sts/semeval-sts/all/2015.train.tsv data/sts/semeval-sts/all/2015.val.tsv
"""

from __future__ import print_function
from __future__ import division

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Input, Lambda
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import numpy as np

import pysts.eval as ev
from pysts.kerasts import graph_input_sts
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import STSPearsonCB
from pysts.kerasts.objectives import pearsonobj
import pysts.loader as loader
import pysts.nlp as nlp
from pysts.vocab import Vocabulary

from . import AbstractTask


class STSTask(AbstractTask):
    def __init__(self):
        self.name = 'sts'
        self.spad = 60
        self.s0pad = self.spad
        self.s1pad = self.spad
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['ptscorer'] = B.dot_ptscorer

        c['loss'] = pearsonobj  # ...or 'categorical_crossentropy'
        c['epochs'] = 32

    def load_set(self, fname):
        def load_file(fname, skip_unlabeled=True):
            # XXX: ugly logic
            if 'sick2014' in fname:
                return loader.load_sick2014(fname)
            else:
                return loader.load_sts(fname, skip_unlabeled=skip_unlabeled)
        s0, s1, y = load_file(fname)

        if self.vocab is None:
            vocab = Vocabulary(s0 + s1, prune_N=self.c['embprune'], icase=self.c['embicase'])
        else:
            vocab = self.vocab

        si0, sj0 = vocab.vectorize(s0, self.emb, spad=self.s0pad)
        si1, sj1 = vocab.vectorize(s1, self.emb, spad=self.s1pad)
        f0, f1 = nlp.sentence_flags(s0, s1, self.s0pad, self.s1pad)
        # gr = graph_input_sts(si0, si1, sj0, sj1, y, f0, f1, s0, s1)
        gr = graph_input_sts(si0, si1, sj0, sj1, y, f0, f1)
        return (gr, y, vocab)

    def prep_model(self, module_prep_model):
        # Input embedding and encoding
        # model inputs   
        si0 = Input(name='si0', shape=(self.s0pad,), dtype='int32')
        se0 = Input(name='se0', shape=(self.s0pad,self.emb.N))
        si1 = Input(name='si1', shape=(self.s1pad,), dtype='int32')
        se1 = Input(name='se1', shape=(self.s1pad,self.emb.N))
        inputs = [si0, se0, si1, se1]
        if self.c['e_add_flags']:
            f0 = Input(name='f0', shape=(self.s0pad, nlp.flagsdim))
            f1 = Input(name='f1', shape=(self.s1pad, nlp.flagsdim))
            inputs = [si0, se0, si1, se1, f0, f1]

        # embedding block     
        embedding, N_emb = B.embedding(self.emb, self.vocab, self.s0pad, self.s1pad,
                                       self.c['inp_e_dropout'], self.c['inp_w_dropout'], 
                                       add_flags=self.c['e_add_flags'])
        embedded = embedding(inputs)
        print(embedding.get_output_shape_at(0))
        print(embedded)
        print(N_emb)
        # Sentence-aggregate embeddings

        # model_block = module_prep_model(N_emb, self.s0pad, self.s1pad, self.c)
        # outputs = model_block(embedded)

        TDLayer = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0], ) + shape[2:])
        e0b = TDLayer(embedded[0])
        e1b = TDLayer(embedded[1])
        final_outputs = [e0b, e1b]

        # Measurement
        
        if self.c['ptscorer'] == '1':
            # special scoring mode just based on the answer
            # (assuming that the question match is carried over to the answer
            # via attention or another mechanism)
            ptscorer = B.cat_ptscorer
            final_outputs = [final_outputs[1]]
        else:
            ptscorer = self.c['ptscorer']
        
        kwargs = dict()
        if ptscorer == B.mlp_ptscorer:
            kwargs['sum_mode'] = self.c['mlpsum']
            kwargs['Dinit'] = self.c['Dinit']

        scoreS = Activation('linear')(ptscorer(final_outputs, self.c['Ddim'], N_emb, self.c['l2reg'], **kwargs))

        out = Dense(6, kernel_regularizer=l2(self.c['l2reg']))(scoreS)

        outS = Activation('softmax')(out)
        
        model = Model(inputs=inputs, outputs=outS)
        return model

    def build_model(self, module_prep_model, do_compile=True):
        if self.c['ptscorer'] is None:
            # non-neural model
            return module_prep_model(self.vocab, self.c, output='classes')

        model = self.prep_model(module_prep_model)

        for lname in self.c['fix_layers']:
            model.nodes[lname].trainable = False

        if do_compile:
            model.compile(loss=self.c['loss'], optimizer=self.c['opt'])
        return model

    def fit_callbacks(self, weightsf):
        return [STSPearsonCB(self, self.gr, self.grv),
                ModelCheckpoint(weightsf, save_best_only=True, monitor='pearson', mode='max'),
                EarlyStopping(monitor='pearson', mode='max', patience=3)]

    def predict(self, model, gr):
        batch_size = 3000  # XXX: hardcoded
        ypred = []
        for ogr, _ in self.sample_pairs(gr, batch_size, shuffle=False, once=True):
            ypred += list(model.predict(ogr))
        return np.array(ypred)

    def eval(self, model):
        res = []
        for gr, fname in [(self.gr, self.trainf), (self.grv, self.valf), (self.grt, self.testf)]:
            if gr is None:
                res.append(None)
                continue
            ypred = self.predict(model, gr)
            res.append(ev.eval_sts(ypred, gr['classes'], fname))
        return tuple(res)

    def res_columns(self, mres, pfx=' '):
        """ Produce README-format markdown table row piece summarizing
        important statistics """
        return('%s%.6f |%s%.6f |%s%.6f'
               % (pfx, mres[self.trainf]['Pearson'],
                  pfx, mres[self.valf]['Pearson'],
                  pfx, mres[self.testf].get('Pearson', np.nan)))


def task():
    return STSTask()
