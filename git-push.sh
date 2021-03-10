#!/usr/bin/env bash

# git subtree add --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master --squash

git subtree push --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master
git subtree push --prefix=code/keras_demo keras_demo master
git push
