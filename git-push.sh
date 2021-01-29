#!/usr/bin/env bash

git subtree push --prefix=code/keras_demo/keras4bert bert_by_keras master --squash
git subtree push --prefix=code/keras_demo keras_demo master --squash
git push
