#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-23 19:42
    
Author:
    huayang
    
Subject:
    Bert 模型主要代码
    - 模型没有写成类的形式，虽然通过类可以方便的传输内部变量来减少代码量（主要是参数的传递），并拓展模型：
        - 模型拓展可以参考 bojone/bert4keras/models.py 中 UniLM_Mask 的实现及使用，通过继承该类，使 bert 具有直接训练 seq2seq 的能力；
        - 相关文章：从语言模型到Seq2Seq：Transformer如戏，全靠Mask | https://kexue.fm/archives/6933
    - 但也正因为这一点可能会使 tensor 的流向变得不够清晰，这与学习目的相悖。
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

from bert_by_keras.layers import *
from bert_by_keras.utils import gelu


class _OutputType:
    """模型输出的类型"""
    SENTENCE_EMBEDDING = 'sentence_embedding'
    CLS_EMBEDDING = 'cls_embedding'
    MLM_PROBABILITY = 'mlm_probability'
    NSP_PROBABILITY = 'nsp_probability'

    @staticmethod
    def get_type_set():
        """"""
        return {v for k, v in _OutputType.__dict__.items() if not k.startswith('__') and isinstance(v, str)}


class _LayerName:
    """各层的命名，用于加载 ckpt 和 fine-tune 时获取指定层，减少硬编码"""

    def __init__(self, n_transformer_block):
        """"""
        self.n_transformer_block = n_transformer_block

        # input layer
        self.input_token = 'Input-Token'
        self.input_segment = 'Input-Segment'
        # embedding layer
        self.embed_token = 'Embedding-Token'
        self.embed_segment = 'Embedding-Segment'
        self.embed_add = 'Embedding-Add'
        self.embed_position = 'Embedding-Position'
        self.embed_norm = 'Embedding-Norm'
        self.embed_dropout = 'Embedding-Dropout'

        # transformer block
        self._transform_temp = 'Transformer-{block_index}-{sub_layer}'
        # transform_sub_layer
        self.attn = 'MultiHeadSelfAttention'
        self.attn_dropout = 'MultiHeadSelfAttention-Dropout'
        self.attn_add = 'MultiHeadSelfAttention-Add'
        self.attn_norm = 'MultiHeadSelfAttention-Norm'
        self.ffn = 'FeedForward'
        self.ffn_dropout = 'FeedForward-Dropout'
        self.ffn_add = 'FeedForward-Add'
        self.ffn_norm = 'FeedForward-Norm'

        # output layer
        self.pool = 'Pool'
        self.pool_dense = 'Pool-Dense'
        self.mlm_dense = 'MLM-Dense'
        self.mlm_norm = 'MLM-Norm'
        self.mlm_softmax = 'MLM-Softmax'
        self.nsp_softmax = 'NSP-Softmax'

    def transformer(self, block_index, sub_layer):
        return self._transform_temp.format(block_index=block_index, sub_layer=sub_layer)


def build_bret(config_path=None,
               ckpt_path=None,
               return_config=False,
               return_layer_name=False,
               sequence_len=None):
    """"""

    def _arg_replace(arg_name, arg_name_new):
        if arg_name in config and arg_name != arg_name_new:
            config[arg_name_new] = config[arg_name]
            config.pop(arg_name)

    def _remove_arg(arg_name):
        config.pop(arg_name)

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
    # config.update(kwargs)
    config['sequence_len'] = sequence_len

    model, layer_name = bert(**config, return_layer_name=True)
    if ckpt_path:
        load_model_weights_from_checkpoint(model, config, ckpt_path, layer_name)

    ret = [model]
    if return_config:
        ret.append(config)

    if return_layer_name:
        ret.append(layer_name)

    return model if len(ret) <= 1 else ret


def bert_output_adjust(model,
                       output_type='cls_embedding',
                       training=False, ):
    """"""
    assert output_type in _OutputType.get_type_set()

    # outputs = [sequence_embedding, cls_embedding, mlm_probability, nsp_probability]
    outputs = model.outputs
    if output_type == _OutputType.SENTENCE_EMBEDDING:
        outputs = outputs[0]
    elif output_type == _OutputType.CLS_EMBEDDING:
        outputs = outputs[1]
    elif output_type == _OutputType.MLM_PROBABILITY:
        outputs = outputs[2]
    elif output_type == _OutputType.NSP_PROBABILITY:
        outputs = outputs[3]
    elif training:  # 原始 bert 是一个多任务联合训练模型，包括 MLM 和 NSP，因此有两个输出
        outputs = [outputs[2], outputs[3]]

    model_fix = keras.Model(model.inputs, outputs=outputs, name='Bert_fix')

    return model_fix


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
         initializer=None,
         return_layer_name=False):
    """"""
    # args assert
    embedding_size = embedding_size or n_hidden_unit
    initializer = initializer or keras.initializers.TruncatedNormal(stddev=initializer_range)

    layer_name = _LayerName(n_transformer_block)

    def _check_args():
        # 目前暂不支持 embedding_size != n_hidden_unit
        assert embedding_size == n_hidden_unit

    _check_args()
    # inputs
    inputs = get_inputs(sequence_len, layer_name)

    # flow
    x, embed_weights = apply_embedding_layer(inputs,
                                             vocab_size,
                                             segment_vocab_size,
                                             max_position_len,
                                             embedding_size,
                                             dropout_rate,
                                             layer_name)

    for block_index in range(n_transformer_block):
        x = apply_transformer_block(x,
                                    block_index,
                                    n_attention_head,
                                    n_unit_each_head,
                                    n_hidden_unit,
                                    attention_dropout_rate,
                                    n_intermediate_unit,
                                    hidden_activation,
                                    initializer,
                                    layer_name)

    outputs = apply_output_layer(x,
                                 embed_weights,
                                 n_hidden_unit,
                                 hidden_activation,
                                 initializer,
                                 layer_name)

    model = keras.Model(inputs, outputs, name='Bert')

    ret = [model]
    if return_layer_name:
        ret.append(layer_name)

    return model if len(ret) <= 1 else ret


def get_inputs(sequence_len, layer_name):
    """"""
    x_in = keras.layers.Input(shape=(sequence_len,), name=layer_name.input_token)
    s_in = keras.layers.Input(shape=(sequence_len,), name=layer_name.input_segment)

    inputs = [x_in, s_in]
    return inputs


def apply_embedding_layer(inputs,
                          vocab_size,
                          segment_vocab_size,
                          max_sequence_len,
                          embedding_size,
                          dropout_rate,
                          layer_name: _LayerName):
    """"""
    x, s = inputs
    # embed_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
    #                                      name='Embedding-Token')
    # embed_weights = embed_layer.embeddings  # 不能直接获取
    x, embed_weights = CustomEmbedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True,
                                       name=layer_name.embed_token)(x)
    s = keras.layers.Embedding(input_dim=segment_vocab_size, output_dim=embedding_size,
                               name=layer_name.embed_segment)(s)

    x = CustomAdd(name=layer_name.embed_add)([x, s])  # [x, s] 的顺序不能变
    x = PositionEmbedding(input_dim=max_sequence_len, output_dim=embedding_size, name=layer_name.embed_position)(x)
    x = LayerNormalization(name=layer_name.embed_norm)(x)
    x = keras.layers.Dropout(dropout_rate, name=layer_name.embed_dropout)(x)

    return x, embed_weights


def apply_transformer_block(inputs,
                            block_index,
                            n_attention_head,
                            n_unit_each_head,
                            n_hidden_unit,
                            dropout_rate,
                            n_intermediate_unit,
                            hidden_act,
                            initializer,
                            layer_name: _LayerName):
    """Attn -> Drop -> Add -> LN -> FFN -> Drop -> Add -> LN"""
    x = inputs
    xi = x
    x = MultiHeadAttention(n_unit=n_hidden_unit,
                           n_head=n_attention_head,
                           n_unit_each_head=n_unit_each_head,
                           name=layer_name.transformer(block_index, layer_name.attn))([x, x, x])
    x = keras.layers.Dropout(dropout_rate, name=layer_name.transformer(block_index, layer_name.attn_dropout))(x)
    x = keras.layers.Add(name=layer_name.transformer(block_index, layer_name.attn_add))([xi, x])
    x = LayerNormalization(name=layer_name.transformer(block_index, layer_name.attn_norm))(x)

    xi = x
    x = FeedForward(units=n_intermediate_unit,
                    activation=hidden_act,
                    kernel_initializer=initializer,
                    name=layer_name.transformer(block_index, layer_name.ffn))(x)
    x = keras.layers.Dropout(dropout_rate, name=layer_name.transformer(block_index, layer_name.ffn_dropout))(x)
    x = keras.layers.Add(name=layer_name.transformer(block_index, layer_name.ffn_add))([xi, x])
    x = LayerNormalization(name=layer_name.transformer(block_index, layer_name.ffn_norm))(x)

    return x


def apply_output_layer(inputs,
                       embed_weights,
                       n_hidden_unit,
                       hidden_activation,
                       initializer,
                       layer_name):
    """"""
    # 可能包含多个输出
    outputs = []

    x = inputs
    # sequence_embedding
    outputs.append(x)

    # cls_embedding
    x = outputs[0]  # sequence_embedding
    x = keras.layers.Lambda(function=lambda tensor: tensor[:, 0], name=layer_name.pool)(x)  # 提取 [CLS] embedding
    x = keras.layers.Dense(units=n_hidden_unit, activation='tanh', kernel_initializer=initializer,
                           name=layer_name.pool_dense)(x)
    outputs.append(x)  # cls_embedding

    # mlm_probability (Task 1)
    x = outputs[0]  # sequence_embedding
    x = keras.layers.Dense(units=n_hidden_unit, activation=hidden_activation, name=layer_name.mlm_dense)(x)
    x = LayerNormalization(name=layer_name.mlm_norm)(x)
    x = EmbeddingSimilarity(name=layer_name.mlm_softmax)([x, embed_weights])
    outputs.append(x)  # mlm_probability

    # nsp_probability (Task 2)
    x = outputs[1]  # cls_embedding
    x = keras.layers.Dense(units=2, activation='softmax', kernel_initializer=initializer,
                           name=layer_name.nsp_softmax)(x)
    outputs.append(x)  # nsp_probability

    return outputs  # [sequence_embedding, cls_embedding, mlm_probability, nsp_probability]


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       layer_name: _LayerName):
    """Load trained official model from checkpoint.
    """
    _loader = lambda name: tf.train.load_variable(checkpoint_file, name)

    model.get_layer(name=layer_name.embed_token).set_weights([
        _loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name=layer_name.embed_segment).set_weights([
        _loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name=layer_name.embed_position).set_weights([
        _loader('bert/embeddings/position_embeddings')[:config['max_position_len'], :],
    ])
    model.get_layer(name=layer_name.embed_norm).set_weights([
        _loader('bert/embeddings/LayerNorm/gamma'),
        _loader('bert/embeddings/LayerNorm/beta'),
    ])

    for block_index in range(config['n_transformer_block']):
        weight_prefix = 'bert/encoder/layer_%d/' % block_index
        model.get_layer(name=layer_name.transformer(block_index, layer_name.attn)).set_weights([
            _loader(weight_prefix + 'attention/self/query/kernel'),
            _loader(weight_prefix + 'attention/self/query/bias'),
            _loader(weight_prefix + 'attention/self/key/kernel'),
            _loader(weight_prefix + 'attention/self/key/bias'),
            _loader(weight_prefix + 'attention/self/value/kernel'),
            _loader(weight_prefix + 'attention/self/value/bias'),
            _loader(weight_prefix + 'attention/output/dense/kernel'),
            _loader(weight_prefix + 'attention/output/dense/bias'),
        ])
        model.get_layer(name=layer_name.transformer(block_index, layer_name.attn_norm)).set_weights([
            _loader(weight_prefix + 'attention/output/LayerNorm/gamma'),
            _loader(weight_prefix + 'attention/output/LayerNorm/beta'),
        ])
        model.get_layer(name=layer_name.transformer(block_index, layer_name.ffn)).set_weights([
            _loader(weight_prefix + 'intermediate/dense/kernel'),
            _loader(weight_prefix + 'intermediate/dense/bias'),
            _loader(weight_prefix + 'output/dense/kernel'),
            _loader(weight_prefix + 'output/dense/bias'),
        ])
        model.get_layer(name=layer_name.transformer(block_index, layer_name.ffn_norm)).set_weights([
            _loader(weight_prefix + 'output/LayerNorm/gamma'),
            _loader(weight_prefix + 'output/LayerNorm/beta'),
        ])

    model.get_layer(name=layer_name.pool_dense).set_weights([
        _loader('bert/pooler/dense/kernel'),
        _loader('bert/pooler/dense/bias'),
    ])
    model.get_layer(name=layer_name.mlm_dense).set_weights([
        _loader('cls/predictions/transform/dense/kernel'),
        _loader('cls/predictions/transform/dense/bias'),
    ])
    model.get_layer(name=layer_name.mlm_norm).set_weights([
        _loader('cls/predictions/transform/LayerNorm/gamma'),
        _loader('cls/predictions/transform/LayerNorm/beta'),
    ])
    model.get_layer(name=layer_name.mlm_softmax).set_weights([
        _loader('cls/predictions/output_bias'),
    ])
    model.get_layer(name=layer_name.nsp_softmax).set_weights([
        np.transpose(_loader('cls/seq_relationship/output_weights')),
        _loader('cls/seq_relationship/output_bias'),
    ])


def model_fine_tune_config(model: keras.Model,
                           layer_name: _LayerName,
                           is_fine_tune=True,
                           is_fine_tune_norm=False,
                           is_fine_tune_embedding=False,
                           n_fine_tune_transformer_block=None):
    """
    模型 fine tune 配置，如果有其他需求可以参考这个方法修改：
        - 比如 output 部分一般都是要参与训练的，这个应该没疑问，就没写相关控制方法；
        - 比如像控制特定哪几个 transformer 要参与训练
        
    Args:
        model: 
        layer_name: 
        is_fine_tune: 是否微调模型，默认 True
        is_fine_tune_norm: 是否微调 LayerNormalization 层，默认 False
        is_fine_tune_embedding: 是否微调 Embedding，默认 False
        n_fine_tune_transformer_block: 控制微调的 transformer_block 数量，默认全部参与微调
            - 优先参与训练的是接近 output 的 block；
            - block 内部的层要么全部参与微调，要么都不；

    Returns:
        None
    """
    model.trainable = True
    if not is_fine_tune:
        model.trainable = False
        return model

    if n_fine_tune_transformer_block and n_fine_tune_transformer_block <= layer_name.n_transformer_block:
        # 相当于把剩下的 block 调整的不可训练
        for block_index in range(layer_name.n_transformer_block - n_fine_tune_transformer_block):
            model.get_layer(layer_name.transformer(block_index, layer_name.attn)).trainable = False
            model.get_layer(layer_name.transformer(block_index, layer_name.ffn)).trainable = False

    if not is_fine_tune_norm:
        for layer in model.layers:
            if isinstance(layer, LayerNormalization):
                layer.trainable = False

    if not is_fine_tune_embedding:
        model.get_layer(layer_name.embed_token).trainable = False
        model.get_layer(layer_name.embed_segment).trainable = False
        model.get_layer(layer_name.embed_position).trainable = False

    return model
