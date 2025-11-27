# Loss weighting strategies.
from .dwa import DWALossWeighter
from .gradnorm import GradNormWeighter
from .kend_gal import KendallGalWeighter

__all__ = ["DWALossWeighter", "GradNormWeighter", "KendallGalWeighter"]
