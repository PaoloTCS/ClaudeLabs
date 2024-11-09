# src/claudelabs/__init__.py

"""
ClaudeLabs - Knowledge Framework for tensor-based compression and symmetry detection
"""

__version__ = "0.1.0"
__author__ = "Paolo Pignatelli"
__email__ = "your.email@example.com"

from . import core
from . import analysis
from . import knowledge
from . import utils

# Make key classes available at package level
from .core.data_structures import (
    CompressionUnit,
    TensorPosition,
    SymmetryMetrics,
    CompressionType,
    RelationshipType
)

from .analysis.network_analysis import NetworkAnalysisIntegrator
from .analysis.transition_detection import TransitionDetector

from .knowledge.intention import IntentionSpace, IntentState
from .knowledge.curiosity import CuriosityCrawler

# Version information
VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'release': 'alpha'
}

def get_version():
    """Return the version string."""
    return __version__