#!/usr/bin/env bash

git push
git subtree push --prefix=code/keras_demo keras_demo master --squash
git subtree push --prefix=code/bert_keras bert_keras master --squash