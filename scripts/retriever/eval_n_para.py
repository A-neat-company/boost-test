#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os
import numpy as np
import regex

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils

from question_classifier.input_example import InputExample
from transformers import BertTokenizer
# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def reconstruct_with_max_seq(doc, max_seq, tokenizer):
    ret = []
    count = 0
    to_add = []
    len_to_add = 0
    for split in regex.split(r'\n+', doc) :
        split = split.strip()
        if len(split) == 0:
            continue
    
        len_split = len(tokenizer.tokenize(split))
        if len(to_add) > 0 and len_to_add + len_split > max_seq:
            to_add = []
            len_to_add = 0
            count+=1
        
        to_add.append(split)
        len_to_add += len_split

    if len(to_add) > 0:
        count+=1

    return count

def get_has_answer(answer_doc, match, PROCESS_DB, PROCESS_TOK, tokenizer):
    """Search through all the top docs to see if they have the answer."""
    answer, doc_ids, _ = answer_doc
    doc_ids = doc_ids[0]
    n_paras = 0
    for doc_id in doc_ids:
        text = PROCESS_DB.get_doc_text(doc_id)
        n_paras+=reconstruct_with_max_seq(text, 384, tokenizer)
    return n_paras

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()



    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []

    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)
    
    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
        questions, k=args.n_docs, num_workers=args.num_workers
    )
    ranker = []

    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    PROCESS_TOK = tok_class(**tok_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

    #answers_docs = rerankDocs(questions, answers, closest_docs, PROCESS_DB)
    answers_docs = zip(answers, closest_docs, questions)

    logger.info('Retrieving texts and computing scores...')
    has_answers = []

    
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
    amplified_Dataset = []
    lens = []
    for answer_doc in answers_docs:
        paras = (get_has_answer(answer_doc, args.match, PROCESS_DB, PROCESS_TOK, tokenizer))
        lens.append(paras)

    print(np.around(np.mean(lens)))
