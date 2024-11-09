# src/claudelabs/core/data_structures.py

from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional, Tuple
from enum import IntEnum, auto
import numpy as np

class RelationshipType(IntEnum):
    """Types of relationships between knowledge elements"""
    TOKEN_LEVEL = auto()
    PATTERN_LEVEL = auto()
    META_LEVEL = auto()
    RATIONAL_LEVEL = auto()
    SYMMETRY_LEVEL = auto()

class CompressionType(IntEnum):
    """Types of compression operations"""
    SEQUENTIAL = auto()
    STRUCTURAL = auto()
    SEMANTIC = auto()
    META = auto()
    SYMMETRY = auto()
    RATIONAL = auto()

@dataclass
class SymmetryMetrics:
    """Metrics for tracking symmetry patterns"""
    symmetry_score: float = 0.0
    symmetry_type: str = ""
    symmetry_stability: float = 0.0
    transformation_group: Set[str] = field(default_factory=set)
    invariant_properties: Set[str] = field(default_factory=set)

@dataclass
class TensorPosition:
    """Position in knowledge tensor T(L,P,C)"""
    level: int
    position: int
    channel: int
    symmetry_indices: Dict[str, float] = field(default_factory=dict)
    
    def as_tuple(self) -> Tuple[int, int, int]:
        """Return position as a tuple"""
        return (self.level, self.position, self.channel)
    
    def get_symmetry_coordinates(self) -> Dict[str, Tuple[float, ...]]:
        """Get symmetry-based coordinate systems"""
        return {
            'standard': self.as_tuple(),
            'symmetry_weighted': tuple(self.symmetry_indices.values())
        }

@dataclass
class CompressionUnit:
    """Core unit for knowledge compression"""
    tokens: Set[str]
    symbol: str
    tensor_position: TensorPosition
    compression_type: CompressionType
    
    # Network properties
    centrality: float = 0.0
    community_id: Optional[int] = None
    structural_role: Optional[int] = None
    
    # Symmetry properties
    symmetry_metrics: SymmetryMetrics = field(default_factory=SymmetryMetrics)
    symmetry_group: Optional[str] = None
    
    # Validation metrics
    confidence: float = 0.0
    stability: float = 0.0
    symmetry_confidence: float = 0.0
    
    # Relationship tracking
    incoming_edges: Dict[str, float] = field(default_factory=dict)
    outgoing_edges: Dict[str, float] = field(default_factory=dict)
    symmetry_edges: Dict[str, float] = field(default_factory=dict)

    def compute_density(self) -> float:
        """Compute local density around this unit"""
        edge_count = len(self.incoming_edges) + len(self.outgoing_edges)
        possible_edges = len(self.tokens) * 2  # Simplified estimate
        return edge_count / max(possible_edges, 1)

    def get_validation_metrics(self) -> Dict[str, float]:
        """Get all validation metrics"""
        return {
            'confidence': self.confidence,
            'stability': self.stability,
            'symmetry_confidence': self.symmetry_confidence,
            'density': self.compute_density()
        }

    def merge_with(self, other: 'CompressionUnit') -> 'CompressionUnit':
        """Create a new unit by merging with another unit"""
        return CompressionUnit(
            tokens=self.tokens.union(other.tokens),
            symbol=f"{self.symbol}+{other.symbol}",
            tensor_position=TensorPosition(
                level=max(self.tensor_position.level, other.tensor_position.level),
                position=min(self.tensor_position.position, other.tensor_position.position),
                channel=self.tensor_position.channel
            ),
            compression_type=CompressionType.META,
            symmetry_metrics=SymmetryMetrics(
                symmetry_score=(self.symmetry_metrics.symmetry_score + 
                              other.symmetry_metrics.symmetry_score) / 2
            )
        )

@dataclass
class TransitionMetrics:
    """Metrics for tracking transitions in knowledge structure"""
    abstraction_level: float
    self_reference: float
    consistency: float
    independence: float
    stability: float
    generative_power: float
    
    # Symmetry metrics
    symmetry_increase: float = 0.0
    symmetry_stability: float = 0.0
    symmetry_coherence: float = 0.0
    
    # Network metrics
    community_coherence: float = 0.0
    structural_stability: float = 0.0
    centrality_change: float = 0.0
    
    # Knowledge flow metrics
    empirical_rational_flow: float = 0.0
    rational_empirical_flow: float = 0.0
    symmetry_flow: float = 0.0
    
    # Enhanced tensor metrics
    tensor_density: float = 0.0
    channel_distribution: Dict[int, float] = field(default_factory=dict)
    symmetry_distribution: Dict[str, float] = field(default_factory=dict)
    
    def as_vector(self) -> np.ndarray:
        """Convert metrics to feature vector"""
        return np.array([
            self.abstraction_level,
            self.self_reference,
            self.consistency,
            self.independence,
            self.stability,
            self.generative_power,
            self.symmetry_increase,
            self.symmetry_stability,
            self.symmetry_coherence,
            self.community_coherence,
            self.structural_stability,
            self.centrality_change,
            self.empirical_rational_flow,
            self.rational_empirical_flow,
            self.symmetry_flow,
            self.tensor_density
        ])