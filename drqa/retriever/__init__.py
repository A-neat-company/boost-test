#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

import zmq

import cPickle as pickle

class Server(object):
    def __init__(self):
        context = zmq.Context()

        self.receiver = context.socket(zmq.PULL)
        self.receiver.bind("tcp://*:1234")

        self.sender = context.socket(zmq.PUSH)
        self.sender.bind("tcp://*:1235")

    def send(self, data):
        self.sender.send(pickle.dumps(data))

    def recv(self):
        data = self.receiver.recv()
        return pickle.loads(data)

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    ),
    'elastic_url': 'localhost:9200'
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    if name == 'elasticsearch':
        return ElasticDocRanker
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .elastic_doc_ranker import ElasticDocRanker
