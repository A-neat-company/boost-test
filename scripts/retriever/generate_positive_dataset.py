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


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None

def check_has_answer(answer, doc_ids, PROCESS_DB, PROCESS_TOK):

    paragraphs = [utils.normalize(PROCESS_DB.get_doc_text(doc_id)) for doc_id in doc_ids]
    
    has_answ = []
    for paragraph in paragraphs:
        has_answ.append(check_ans(answer, paragraph, PROCESS_TOK))

    return paragraphs, has_answ

def check_ans(answer, paragraph, tokenizer):

    text = tokenizer.tokenize(paragraph).words(uncased=True)
    for single_answer in answer:
        single_answer = utils.normalize(single_answer)
        single_answer = PROCESS_TOK.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
            
        for i in range(0, len(text) - len(single_answer) + 1):
            if  single_answer == text[i: i + len(single_answer)]:
                return 1
    return 0


def get_has_answer(answer_doc, match, PROCESS_DB, PROCESS_TOK):
    """Search through all the top docs to see if they have the answer."""
    answer, doc_ids, _ = answer_doc
    doc_ids = doc_ids[0]
    ret = []
    res = []
    paras, answs = check_has_answer(answer, doc_ids, PROCESS_DB, PROCESS_TOK) 
    if set([1]).issubset(set(answs)):
        pos_indexes = np.where(np.array(answs) == 1)[0]
        ret = [paras[i] for i in pos_indexes]
    return ret


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
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--save-dir', type=str, default=None)
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

    amplified_Dataset = []
    for answer_doc in answers_docs:
        paras = (get_has_answer(answer_doc, args.match, PROCESS_DB, PROCESS_TOK))
        if len(paras) > 0:
            amplified_Dataset.append({
                        "question" : answer_doc[2], 
                        "contexts" : paras, 
                        "answers" : answer_doc[0]})

    print("saving dataset")
    print(len(amplified_Dataset))
    basename = os.path.basename(args.dataset)
    dataset_name, _ = os.path.splitext(basename)
    file_name= 'positive_' + dataset_name + '.json'
    with open(os.path.join(args.save_dir, file_name) , 'w') as fp:
        json.dump(amplified_Dataset, fp)

