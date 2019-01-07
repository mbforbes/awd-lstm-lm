#!/bin/bash

# tbooks
CUDA_VISIBLE_DEVICES=1 python generate.py \
    models/main-tbooks-e_4-b_40000-vppl_70.90.pt \
    data/tbooks/vocab.pkl \
    data/tbooks/test2.initials.txt \
    output/cachelm.tbooks.test2.txt

# trip
CUDA_VISIBLE_DEVICES=1 python generate.py \
    models/main-trip-e_12-b_40000-vppl_29.71.pt \
    data/trip/vocab.pkl \
    data/trip/test2.initials.txt \
    output/cachelm.trip.test2.txt \
    --use-eog

# wiki2
CUDA_VISIBLE_DEVICES=1 python generate.py \
    models/main-wiki2-e_2-b_0-vppl_61.99.pt \
    data/wiki2/vocab.pkl \
    data/wiki2/test2.initials.txt \
    output/cachelm.wiki2.test2.txt \
    --use-eog
