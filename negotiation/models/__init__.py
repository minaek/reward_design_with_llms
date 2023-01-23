from models.dialog_model import DialogModel
from models.cda_rnn_model import CdaRnnModel

MODELS = {
    "rnn_model": DialogModel,
    "cda_rnn_model": CdaRnnModel,
}


def get_model_names():
    return MODELS.keys()


def get_model_type(name):
    return MODELS[name]
