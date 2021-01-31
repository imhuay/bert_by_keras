#!/usr/bin/env bash

git subtree push --prefix=code/keras_utils/keras4bert bert_by_keras master --squash
git subtree push --prefix=code/keras_utils keras_utils master --squash
git push
