#!/usr/bin/env bash

git subtree push --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master --squash
git subtree push --prefix=code/keras_demo/keras_utils keras_utils master --squash
git push
