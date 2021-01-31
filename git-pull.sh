#!/usr/bin/env bash

git pull
git subtree pull --prefix=code/keras_utils keras_utils master --squash
git subtree pull --prefix=code/keras_utils/keras4bert bert_by_keras master --squash
