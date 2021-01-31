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

from .layers import *
from .utils import gelu


class ReturnType:
    SENTENCE_EMBEDDING = 'sentence_embedding'
    CLS_EMBEDDING = 'cls_embedding'
    MLM_PROBABILITY = 'mlm_probability'
    NSP_PROBABILITY = 'nsp_probability'


def build_bret_from_config(config_path=None,
                           ckpt_path=None,
                           return_type='cls_embedding',
                           training=False,
                           return_full_model=False,
                           return_config=False,
                           **kwargs):
    """"""

    def _arg_replace(arg_name, arg_name_new):
        if arg_name in config and arg_name != arg_name_new:
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
    
    # 这几个 remove 的参数还没深入研究是怎么用的
    _remove_arg('directionality')
    _remove_arg('pooler_fc_size')
    _remove_arg('pooler_num_attention_heads')
    _remove_arg('pooler_num_fc_layers')
    _remove_arg('pooler_size_per_head')
    _remove_arg('pooler_type')
    _arg_replace('attention_probs_dropout_prob', 'attention_dropout_rate')
    _arg_replace('hidden_act', 'hidden_activation')
    _arg_replace('hidden_dropout_prob', 'dropout_rate')
    _arg_replace('hidden_size', 'n_hidden_unit')
    _arg_replace('initializer_range', 'initializer_range')
    _arg_replace('intermediate_size', 'n_intermediate_unit')
    _arg_replace('max_position_embeddings', 'max_position_len')
    _arg_replace('num_attention_heads', 'n_attention_head')
    _arg_replace('num_hidden_layers', 'n_transformer_block')
    _arg_replace('type_vocab_size', 'segment_vocab_size')
    _arg_replace('vocab_size', 'vocab_size')
    config.update(kwargs)

    model = bert(**config)
    load_model_weights_from_checkpoint(model, config, ckpt_path)

    # outputs = [sequence_embedding, cls_embedding, mlm_probability, nsp_probability]
    outputs = model.outputs
    if return_type == ReturnType.SENTENCE_EMBEDDING:
        outputs = outputs[0]
    elif return_type == ReturnType.CLS_EMBEDDING:
        outputs = outputs[1]
    elif return_type == ReturnType.MLM_PROBABILITY:
        outputs = outputs[2]
    elif return_type == ReturnType.NSP_PROBABILITY:
        outputs = outputs[3]
    elif training:  # 原始 bert 是一个多任务联合训练模型，包括 MLM 和 NSP，因此有两个输出
        outputs = [outputs[2], outputs[3]]

    model_fix = keras.Model(model.inputs, outputs=outputs, name='Bert_fix')
    
    ret = [model_fix]
    if return_full_model:
        ret.append(model)
    
    if return_config:
        ret.append(config)
    
    return ret if len(ret) > 1 else ret[0]


def bert(n_hidden_unit=768,
         n_transformer_block=12,
         n_attention_head=12,
         n_intermediate_unit=3072,
         vocab_size=21128,
         segment_vocab_size=2,
         max_position_len=512,
         sequence_len=None,
         hidden_activation=gelu,
         n_unit_each_head=None,
         embedding_size=None,
         dropout_rate=0.0,
         attention_dropout_rate=0.0,
         initializer_range=0.02,
         initializer=None):
    """"""
    # args assert
    embedding_size = embedding_size or n_hidden_unit
    initializer = initializer or keras.initializers.TruncatedNormal(stddev=initializer_range)

    def _check_args():
        # 目前暂不支持 embedding_size != n_hidden_unit
        assert embedding_size == n_hidden_unit

    _check_args()
    # inputs
    inputs = get_inputs(sequence_len)

    # flow
    x, embed_weights = apply_embedding_layer(inputs,
                                             vocab_size,
                                             segment_vocab_size,
                                             max_position_len,
                                             embedding_size,
                                             dropout_rate)

    for block_index in range(n_transformer_block):
        x = apply_transformer_block(x,
                                    block_index,
                                    n_attention_head,
                                    n_unit_each_head,
                                    n_hidden_unit,
                                    attention_dropout_rate,
                                    n_intermediate_unit,
                                    hidden_activation,
                                    initializer)

    outputs = apply_output_layer(x,
                                 embed_weights,
                                 n_hidden_unit,
                                 hidden_activation,
                                 initializer)

    # outputs = [sequence_embedding, cls_embedding, mlm_probability, nsp_probability]
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


def apply_embedding_layer(inputs, vocab_size, segment_vocab_size, max_sequence_len, embedding_size, dropout_rate):
    """"""
    inputs = inputs[:]
    x, s = inputs
    # embed_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
    #                                      name='Embedding-Token')
    # embed_weights = embed_layer.embeddings  # 不能直接获取
    x, embed_weights = CustomEmbedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
                                       name='Embedding-Token')(x)
    s = keras.layers.Embedding(input_dim=segment_vocab_size, output_dim=embedding_size, name='Embedding-Segment')(s)

    x = CustomAdd(name='Embedding-Token-Segment')([x, s])  # [x, s] 的顺序不能变
    x = PositionEmbedding(input_dim=max_sequence_len, output_dim=embedding_size, name='Embedding-Position')(x)
    # x = keras.layers.LayerNormalization(name='Embedding-Norm')(x)
    x = LayerNormalization(name='Embedding-Norm')(x)
    x = keras.layers.Dropout(dropout_rate, name='Embedding-Dropout')(x)

    return x, embed_weights


def apply_transformer_block(inputs,
                            block_index,
                            n_attention_head,
                            n_unit_each_head,
                            n_hidden_unit,
                            dropout_rate,
                            n_intermediate_unit,
                            hidden_act,
                            initializer):
    """Att --> Add --> LN --> FFN --> Add --> LN"""
    attention_name = 'Transformer-%d-MultiHeadSelfAttention' % block_index
    feed_forward_name = 'Transformer-%d-FeedForward' % block_index

    x = inputs
    xi = x
    x = MultiHeadAttention(n_unit=n_hidden_unit,
                           n_head=n_attention_head,
                           n_unit_each_head=n_unit_each_head,
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


def apply_output_layer(inputs,
                       embed_weights,
                       n_hidden_unit,
                       hidden_activation,
                       initializer):
    """"""
    # 可能包含多个输出
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
    x = keras.layers.Dense(units=n_hidden_unit, activation=hidden_activation, name='MLM-Dense')(x)
    x = LayerNormalization(name='MLM-Norm')(x)
    x = EmbeddingSimilarity(name='MLM-Softmax')([x, embed_weights])
    outputs.append(x)  # mlm softmax

    # Task2: Next Sentence
    x = outputs[1]  # [CLS] 向量
    x = keras.layers.Dense(units=2, activation='softmax', kernel_initializer=initializer, name='NSP-Softmax')(x)
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
    model.get_layer(name='MLM-Softmax').set_weights([
        _loader('cls/predictions/output_bias'),
    ])
    model.get_layer(name='NSP-Softmax').set_weights([
        np.transpose(_loader('cls/seq_relationship/output_weights')),
        _loader('cls/seq_relationship/output_bias'),
    ])
