REGISTRY = {}

from .hpn_rnn_agent import HPN_RNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .asn_rnn_agent import AsnRNNAgent
from .deepset_hyper_rnn_agent import DeepSetHyperRNNAgent
from .deepset_rnn_agent import DeepSetRNNAgent
from .gnn_rnn_agent import GnnRNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_agent import RNNAgent
from .updet_agent import UPDeT

from .attrpe_agent import AttRPEAgent
from .spe_rnn_agent import SPE_RNNAgent
from .spe_light_rnn_agent import SPE_Light_RNNAgent
from .sattpe_rnn_agent import SAttPE_RNNAgent
from .sattpe1_rnn_agent import SAttPE1_RNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["hpn_rnn"] = HPN_RNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["deepset_rnn"] = DeepSetRNNAgent
REGISTRY["deepset_hyper_rnn"] = DeepSetHyperRNNAgent
REGISTRY["updet_agent"] = UPDeT
REGISTRY["asn_rnn"] = AsnRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent

REGISTRY["attrpe"] = AttRPEAgent
REGISTRY["sper"] = SPE_RNNAgent
REGISTRY["sper_light"] = SPE_Light_RNNAgent
REGISTRY["sattper"] = SAttPE_RNNAgent
REGISTRY["sattpe1r"] = SAttPE1_RNNAgent
