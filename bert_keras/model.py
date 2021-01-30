#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-23 19:42
    
Author:
    huayang
    
Subject:
    
"""
import json

import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from .layers import PositionEmbedding, LayerNormalization, MultiHeadAttention, FeedForward, EmbeddingSimilarity, \
    CustomEmbedding
from .utils import gelu


class ReturnType:
    SENTENCE_EMBEDDING = 'sentence_embedding'
    CLS_EMBEDDING = 'cls_embedding'
    NSP_PROBABILITY = 'nsp_probability'
    MLM_PROBABILITY = 'mlm_probability'


def build_bret_from_config(config_path=None,
                           ckpt_path=None,
                           return_type='sentence_embedding',
                           training=False,
                           return_full_model=False,
                           model_summary=False,
                           **kwargs):
    """"""

    def _arg_replace(arg_name, arg_name_new):
        if arg_name in config:
            config[arg_name_new] = config[arg_name]
            config.pop(arg_name)

    def _remove_arg(arg_name):
        config.pop(arg_name)

    def _check_args():
        assert return_type in {ReturnType.SENTENCE_EMBEDDING,
                               ReturnType.CLS_EMBEDDING,
                               ReturnType.MLM_PROBABILITY,
                               ReturnType.NSP_PROBABILITY}

    _check_args()
    config = {}
    if config_path is not None:
        config.update(json.load(open(config_path)))

    _remove_arg('directionality')
    _arg_replace('hidden_dropout_prob', 'dropout_rate')
    _arg_replace('type_vocab_size', 'segment_vocab_size')
    _arg_replace('num_attention_heads', 'n_attention_head')
    _arg_replace('num_hidden_layers', 'n_transformer_block')
    _arg_replace('intermediate_size', 'n_intermediate_unit')
    _arg_replace('hidden_size', 'n_hidden_unit')
    _arg_replace('attention_probs_dropout_prob', 'attention_dropout_rate')
    _arg_replace('max_position_embeddings', 'max_position_len')
    config.update(kwargs)

    model = build_bert(**config)
    load_model_weights_from_checkpoint(model, config, ckpt_path)

    # outputs = [sequecen embedding, cls embedding, mlm softmax, nsp softmax]
    outputs = model.outputs
    if return_type == ReturnType.SENTENCE_EMBEDDING:
        outputs = outputs[0]
    elif return_type == ReturnType.CLS_EMBEDDING:
        outputs = outputs[1]
    elif return_type == ReturnType.MLM_PROBABILITY:
        outputs = outputs[2]
    elif return_type == ReturnType.NSP_PROBABILITY:
        outputs = outputs[3]
    elif training:
        outputs = [outputs[2], outputs[3]]

    if model_summary:
        model.summary(line_length=200)

    model_fix = keras.Model(model.inputs, outputs=outputs, name='Bert_fix')
    
    if return_full_model:
        return model_fix, model
    
    return model_fix


def build_bert(n_hidden_unit=768,
               n_transformer_block=12,
               n_attention_head=12,
               n_intermediate_unit=3072,
               vocab_size=21128,
               segment_vocab_size=2,
               max_position_len=512,
               sequence_len=None,
               hidden_act=gelu,
               n_each_head_unit=None,
               embedding_size=None,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               initializer_range=0.02,
               initializer=None,
               **kwargs):
    """"""
    # args assert
    embedding_size = embedding_size or n_hidden_unit
    n_each_head_unit = n_each_head_unit or n_hidden_unit // n_attention_head
    initializer = initializer or keras.initializers.TruncatedNormal(stddev=initializer_range)

    def _check_args():
        # 目前暂不支持 embedding_size != n_hidden_unit
        assert embedding_size == n_hidden_unit

    _check_args()
    # inputs
    inputs = get_inputs(sequence_len)

    # flow
    x, embed_weights = apply_embeddings(inputs,
                                        vocab_size,
                                        segment_vocab_size,
                                        max_position_len,
                                        embedding_size,
                                        dropout_rate)

    for layer_index in range(n_transformer_block):
        x = apply_main_layers(x,
                              layer_index,
                              n_attention_head,
                              n_each_head_unit,
                              n_hidden_unit,
                              attention_dropout_rate,
                              n_intermediate_unit,
                              hidden_act,
                              initializer)

    outputs = apply_final_layers(x,
                                 embed_weights,
                                 n_hidden_unit,
                                 hidden_act,
                                 initializer)

    # outputs = [sequecen embedding, cls embedding, mlm softmax, nsp softmax]
    # if return_type == ReturnType.SENTENCE_EMBEDDING:
    #     outputs = outputs[0]
    # elif return_type == ReturnType.CLS_EMBEDDING:
    #     outputs = outputs[1]
    # elif return_type == ReturnType.MLM_PROBABILITY:
    #     outputs = outputs[2]
    # elif return_type == ReturnType.NSP_PROBABILITY:
    #     outputs = outputs[3]
    # elif training:
    #     outputs = [outputs[2], outputs[3]]

    model = keras.Model(inputs, outputs, name='Bert')
    return model


def get_inputs(sequence_len):
    """"""
    x_in = keras.layers.Input(shape=(sequence_len,), name='Input-Token')
    s_in = keras.layers.Input(shape=(sequence_len,), name='Input-Segment')

    inputs = [x_in, s_in]
    return inputs


def apply_embeddings(inputs, vocab_size, segment_vocab_size, max_sequence_len, embedding_size, dropout_rate):
    """"""
    inputs = inputs[:]
    x, s = inputs
    # embed_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
    #                                      name='Embedding-Token')
    # embed_weights = embed_layer.embeddings
    x, embed_weights = CustomEmbedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
                                       name='Embedding-Token')(x)
    s = keras.layers.Embedding(input_dim=segment_vocab_size, output_dim=embedding_size, name='Embedding-Segment')(s)

    x = keras.layers.Add(name='Embedding-Token-Segment')([x, s])
    x = PositionEmbedding(input_dim=max_sequence_len, output_dim=embedding_size, name='Embedding-Position')(x)
    # x = keras.layers.LayerNormalization(name='Embedding-Norm')(x)
    x = LayerNormalization(name='Embedding-Norm')(x)
    x = keras.layers.Dropout(dropout_rate, name='Embedding-Dropout')(x)

    return x, embed_weights


def apply_main_layers(inputs,
                      layer_index,
                      n_attention_head,
                      n_each_head_unit,
                      n_hidden_unit,
                      dropout_rate,
                      n_intermediate_unit,
                      hidden_act,
                      initializer):
    """Att --> Add --> LN --> FFN --> Add --> LN"""
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % layer_index
    feed_forward_name = 'Transformer-%d-FeedForward' % layer_index

    x = inputs
    xi = x
    x = MultiHeadAttention(heads=n_attention_head,
                           head_size=n_each_head_unit,
                           out_dim=n_hidden_unit,
                           name=attention_name)([x, x, x])
    x = keras.layers.Dropout(dropout_rate, name='%s-Dropout' % attention_name)(x)
    x = keras.layers.Add(name='%s-Add' % attention_name)([xi, x])
    # x = keras.layers.LayerNormalization(name='%s-Norm' % attention_name)(x)
    x = LayerNormalization(name='%s-Norm' % attention_name)(x)

    xi = x
    x = FeedForward(units=n_intermediate_unit, activation=hidden_act,
                    kernel_initializer=initializer, name=feed_forward_name)(x)
    x = keras.layers.Dropout(dropout_rate, name='%s-Dropout' % feed_forward_name)(x)
    x = keras.layers.Add(name='%s-Add' % feed_forward_name)([xi, x])
    # x = keras.layers.LayerNormalization(name='%s-Norm' % feed_forward_name)(x)
    x = LayerNormalization(name='%s-Norm' % feed_forward_name)(x)

    return x


def apply_final_layers(inputs,
                       embed_weights,
                       n_hidden_unit,
                       hidden_act,
                       initializer):
    """"""
    # 输出可能包含多个部分
    outputs = []

    x = inputs
    outputs.append(x)  # sentence embedding

    # 提取 [CLS] 向量
    x = outputs[0]  # sentence embedding
    x = keras.layers.Lambda(function=lambda tensor: tensor[:, 0], name='Pooler')(x)  # 提取 [CLS] embedding
    x = keras.layers.Dense(units=n_hidden_unit, activation='tanh', kernel_initializer=initializer,
                           name='Pooler-Dense')(x)
    outputs.append(x)  # [CLS] 向量

    # Task1: Masked Language
    x = outputs[0]  # sentence embedding
    x = keras.layers.Dense(units=n_hidden_unit, activation=hidden_act, name='MLM-Dense')(x)
    x = LayerNormalization(name='MLM-Norm')(x)
    x = EmbeddingSimilarity(name='MLM-Sim')([x, embed_weights])
    outputs.append(x)  # mlm softmax

    # Task2: Next Sentence
    x = outputs[1]  # [CLS] 向量
    x = keras.layers.Dense(units=2, activation='softmax', kernel_initializer=initializer, name='NSP-Proba')(x)
    outputs.append(x)  # nsp softmax

    return outputs  # [sequecen embedding, cls embedding, mlm softmax, nsp softmax]


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       training=False):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    """
    _loader = lambda name: tf.train.load_variable(checkpoint_file, name)

    model.get_layer(name='Embedding-Token').set_weights([
        _loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        _loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        _loader('bert/embeddings/position_embeddings')[:config['max_position_len'], :],
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        _loader('bert/embeddings/LayerNorm/gamma'),
        _loader('bert/embeddings/LayerNorm/beta'),
    ])

    for i in range(config['n_transformer_block']):
        layer_prefix = 'Transformer-%d-' % i
        weight_prefix = 'bert/encoder/layer_%d/' % i
        model.get_layer(name=layer_prefix + 'MultiHeadSelfAttention').set_weights([
            _loader(weight_prefix + 'attention/self/query/kernel'),
            _loader(weight_prefix + 'attention/self/query/bias'),
            _loader(weight_prefix + 'attention/self/key/kernel'),
            _loader(weight_prefix + 'attention/self/key/bias'),
            _loader(weight_prefix + 'attention/self/value/kernel'),
            _loader(weight_prefix + 'attention/self/value/bias'),
            _loader(weight_prefix + 'attention/output/dense/kernel'),
            _loader(weight_prefix + 'attention/output/dense/bias'),
        ])
        model.get_layer(name=layer_prefix + 'MultiHeadSelfAttention-Norm').set_weights([
            _loader(weight_prefix + 'attention/output/LayerNorm/gamma'),
            _loader(weight_prefix + 'attention/output/LayerNorm/beta'),
        ])
        model.get_layer(name=layer_prefix + 'FeedForward').set_weights([
            _loader(weight_prefix + 'intermediate/dense/kernel'),
            _loader(weight_prefix + 'intermediate/dense/bias'),
            _loader(weight_prefix + 'output/dense/kernel'),
            _loader(weight_prefix + 'output/dense/bias'),
        ])
        model.get_layer(name=layer_prefix + 'FeedForward-Norm').set_weights([
            _loader(weight_prefix + 'output/LayerNorm/gamma'),
            _loader(weight_prefix + 'output/LayerNorm/beta'),
        ])

    model.get_layer(name='Pooler-Dense').set_weights([
        _loader('bert/pooler/dense/kernel'),
        _loader('bert/pooler/dense/bias'),
    ])
    model.get_layer(name='MLM-Dense').set_weights([
        _loader('cls/predictions/transform/dense/kernel'),
        _loader('cls/predictions/transform/dense/bias'),
    ])
    model.get_layer(name='MLM-Norm').set_weights([
        _loader('cls/predictions/transform/LayerNorm/gamma'),
        _loader('cls/predictions/transform/LayerNorm/beta'),
    ])
    model.get_layer(name='MLM-Sim').set_weights([
        _loader('cls/predictions/output_bias'),
    ])
    model.get_layer(name='NSP-Proba').set_weights([
        np.transpose(_loader('cls/seq_relationship/output_weights')),
        _loader('cls/seq_relationship/output_bias'),
    ])
