#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-27 11:53
    
Author:
    huayang
    
Subject:
    Bert 原生分词器，移除了兼容 python2 的内容

References:
    https://github.com/google-research/bert/blob/master/tokenization.py
"""
import os
import collections
import unicodedata


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


# def convert_tokens_to_ids(vocab, tokens):
#     return convert_by_vocab(vocab, tokens)
#
#
# def convert_ids_to_tokens(inv_vocab, ids):
#     return convert_by_vocab(inv_vocab, ids)


def load_vocab(vocab_file, encoding='utf8'):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, encoding=encoding) as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def tokenize(text, do_lower_case=True):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = _clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = _tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
        if do_lower_case:
            token = token.lower()
            token = _run_strip_accents(token)
        split_tokens.extend(_run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text.

    Examples:
        text = '我爱python，我爱编程；I love python, I like programming.'
        ret = whitespace_tokenize(text)
        # ['我爱python，我爱编程；I', 'love', 'python,', 'I', 'like', 'programming.']
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def _run_strip_accents(text):
    """Strips accents from a piece of text.

    Examples:
        text = 'âbĉ'
        ret = _run_strip_accents(text) # abc
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def _run_split_on_punc(text):
    """Splits punctuation on a piece of text.

    Examples:
        text = '我爱python，我爱编程；I love python, I like programming.'
        ret = _run_split_on_punc(text)
        # ['我爱python', '，', '我爱编程', '；', 'I love python', ',', ' I like programming', '.']
    """
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F)):
        return True

    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class WordPieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class Tokenizer(object):
    """End-to-end tokenizer, containing word-piece_tokenizer."""

    def __init__(self, vocab_file,
                 do_lower_case=True,
                 token_cls='[CLS]',
                 token_sep='[SEP]',
                 token_unk='[UNK]',
                 token_mask='[MASK]',
                 pad_index=0,
                 verbose=0):
        self.vocab = load_vocab(vocab_file)
        if verbose > 0:
            print('Vocab size=%d' % len(self.vocab))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenize = lambda text: tokenize(text, do_lower_case)
        self.word_piece_tokenize = WordPieceTokenizer(vocab=self.vocab).tokenize
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._pad_index = pad_index
        self._token_mask = token_mask

    def encode(self, txt1, txt2=None, max_len=None):
        tokens_txt1 = self.tokenize(txt1)
        tokens_txt2 = self.tokenize(txt2) if txt2 is not None else None
        self._truncate(tokens_txt1, tokens_txt2, max_len)
        tokens, len_txt1, len_txt2 = self._pack(tokens_txt1, tokens_txt2)

        tokens_id = self._convert_tokens_to_ids(tokens)
        segments_id = [0] * len_txt1 + [1] * len_txt2

        if max_len is not None:
            pad_len = max_len - len_txt1 - len_txt2
            tokens_id += [self._pad_index] * pad_len
            segments_id += [0] * pad_len

        return tokens_id, segments_id

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenize(text):
            for sub_token in self.word_piece_tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def _convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def _convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    @property
    def mask_id(self):
        return self._convert_tokens_to_ids([self._token_mask])[0]

    def _pack(self, tokens_1st, tokens_2nd=None):
        packed_tokens_1st = [self._token_cls] + tokens_1st + [self._token_sep]
        if tokens_2nd is not None:
            packed_tokens_2nd = tokens_2nd + [self._token_sep]
            return packed_tokens_1st + packed_tokens_2nd, len(packed_tokens_1st), len(packed_tokens_2nd)
        else:
            return packed_tokens_1st, len(packed_tokens_1st), 0

    @staticmethod
    def _truncate(tokens_1st, tokens_2nd, max_len):
        """"""
        if max_len is None:
            return

        if tokens_2nd is not None:
            while True:
                total_len = len(tokens_1st) + len(tokens_2nd)
                if total_len <= max_len - 3:  # 3 for [CLS] .. tokens_a .. [SEP] .. tokens_b [SEP]
                    break
                if len(tokens_1st) > len(tokens_2nd):
                    tokens_1st.pop()
                else:
                    tokens_2nd.pop()
        else:
            del tokens_1st[max_len - 2:]  # 2 for [CLS] .. tokens .. [SEP]


# 不是单例
# def get_tokenizer(vocab_file=None, **kwargs):
#     """
#
#     Args:
#         vocab_file:
#
#     Returns:
#
#     """
#     if vocab_file is None:
#         pwd = os.path.dirname(__file__)
#         vocab_file = os.path.join(pwd, '../data/vocab/vocab_21128.txt')
#
#     tokenizer = Tokenizer(vocab_file, **kwargs)
#     return tokenizer


# 模块内的变量默认为单例模式
tokenizer = Tokenizer(os.path.join(os.path.dirname(__file__), '../data/vocab/vocab_21128.txt'))

if __name__ == '__main__':
    """"""
    text = '我爱python，我爱编程；I love python, I like programming. Some unk word unaffable'
    # ret = tokenize(text)
    # ['我爱python，我爱编程；I', 'love', 'python,', 'I', 'like', 'programming.']
    # print(ret)

    # vocab_file = r'../data/vocab/vocab_21128.txt'
    # vocab = load_vocab(vocab_file)
    # print(vocab)

    token_ids, segment_ids = tokenizer.encode('语言模型')
    print(token_ids, segment_ids)
    # model.predict([token_ids, segment_ids]
