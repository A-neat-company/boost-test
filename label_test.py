import zmq
import huggingface_hub
import cPickle as pickle

potato = huggingface_hub.hf_hub_download(None, repo_id="POTATO-AI-REPO", filename="TOMATO-MODEL.chkpt", None)

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
