REGISTRY = {}

from .hpn_controller import HPNMAC
from .basic_controller import BasicMAC
from .n_controller import NMAC
from .updet_controller import UPDETController

from .spe_controller import SPEMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["hpn_mac"] = HPNMAC
REGISTRY["updet_mac"] = UPDETController

REGISTRY["spe_mac"] = SPEMAC
