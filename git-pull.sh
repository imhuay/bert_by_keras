#!/usr/bin/env bash

git subtree pull --prefix=code/bert_keras bert_keras master --squash
git subtree pull --prefix=code/keras_demo keras_demo master --squash
git pull
