#!/usr/bin/env bash

git pull
git subtree pull --prefix=code/keras_demo keras_demo master --squash
git subtree pull --prefix=code/keras_demo/keras4bert bert_by_keras master --squash
