import logging

from .bcs import DirichletBC, LocatorMethod, PressureBC
from .fracstep import FractionalStep_AB_CN
from .function import Projector

logging.basicConfig()
logger = logging.getLogger("oasisx")
logging.captureWarnings(capture=True)


__all__ = [
    "Projector",
    "FractionalStep_AB_CN",
    "DirichletBC",
    "LocatorMethod",
    "PressureBC",
]
