# src/claudelabs/analysis/transition_detection.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from collections import defaultdict

from ..core.data_structures import (
    CompressionUnit, 
    TransitionMetrics,
    SymmetryMetrics
)

@dataclass
class TransitionPoint:
    """Represents a detected transition point"""
    level: int
    type: str
    strength: float
    metrics: TransitionMetrics
    precursors: List[str]
    effects: List[str]
    
    def as_dict(self) -> Dict:
        """Convert transition point to dictionary representation"""
        return {
            'level': self.level,
            'type': self.type,
            'strength': self.strength,
            'metrics': self.metrics.__dict__,
            'precursors': self.precursors,
            'effects': self.effects
        }

class TransitionDetector:
    """Detailed analysis of relationship transitions"""
    
    def __init__(self):
        self.transition_history = []
        self.metric_history = defaultdict(list)  # level -> [metrics]
        self.threshold = 0.7
        self.min_pattern_support = 3  # Minimum instances to confirm a pattern
        self.window_size = 3  # Levels to look back for trend analysis
        
    def analyze_transition(self, 
                         relationships: List[CompressionUnit],
                         level: int) -> Optional[TransitionPoint]:
        """
        Analyze relationships for transition points and characteristics
        Returns TransitionPoint if transition detected, None otherwise
        """
        # Calculate current metrics
        metrics = self._calculate_metrics(relationships)
        self._update_history(metrics, level)
        
        # Check for transition
        if self._is_transition_point(metrics, level):
            transition = self._characterize_transition(relationships, metrics, level)
            self.transition_history.append(transition)
            return transition
        
        return None
    
    def _calculate_metrics(self, 
                         relationships: List[CompressionUnit]) -> TransitionMetrics:
        """Calculate detailed metrics for transition detection"""
        return TransitionMetrics(
            abstraction_level=self._measure_abstraction(relationships),
            self_reference=self._measure_self_reference(relationships),
            consistency=self._measure_consistency(relationships),
            independence=self._measure_independence(relationships),
            stability=self._measure_stability(relationships),
            generative_power=self._measure_generative_power(relationships),
            symmetry_increase=self._measure_symmetry_increase(relationships),
            symmetry_stability=self._measure_symmetry_stability(relationships),
            symmetry_coherence=self._measure_symmetry_coherence(relationships)
        )
    
    def _measure_abstraction(self, relationships: List[CompressionUnit]) -> float:
        """
        Measure degree of abstraction in relationships
        Focuses on distance from concrete tokens
        """
        if not relationships:
            return 0.0
            
        total_level = sum(r.tensor_position.level for r in relationships)
        max_level = max(r.tensor_position.level for r in relationships)
        return total_level / (len(relationships) * max_level) if max_level > 0 else 0.0
    
    def _measure_self_reference(self, relationships: List[CompressionUnit]) -> float:
        """
        Measure degree of self-reference in relationship patterns
        Critical for detecting meta-relationship emergence
        """
        self_ref_count = 0
        total_refs = 0
        
        for rel in relationships:
            # Check for symbol appearing in tokens
            if any(rel.symbol in token for token in rel.tokens):
                self_ref_count += 1
            # Check for interconnected references
            if rel.incoming_edges and rel.outgoing_edges:
                common_refs = set(rel.incoming_edges.keys()) & set(rel.outgoing_edges.keys())
                self_ref_count += len(common_refs)
            total_refs += len(rel.incoming_edges) + len(rel.outgoing_edges)
        
        return self_ref_count / (total_refs + 1e-10)
    
    def _measure_consistency(self, relationships: List[CompressionUnit]) -> float:
        """
        Measure internal consistency of relationship system
        Checks for contradictions and mutual support
        """
        consistency_scores = []
        
        for rel1 in relationships:
            rel1_support = set()
            rel1_contradict = set()
            
            for rel2 in relationships:
                if rel1 == rel2:
                    continue
                    
                # Check token overlap
                token_overlap = rel1.tokens & rel2.tokens
                if token_overlap:
                    rel1_support.add(rel2.symbol)
                
                # Check edge consistency
                common_edges = (set(rel1.outgoing_edges.keys()) & 
                              set(rel2.outgoing_edges.keys()))
                for edge in common_edges:
                    if abs(rel1.outgoing_edges[edge] - rel2.outgoing_edges[edge]) > 0.3:
                        rel1_contradict.add(rel2.symbol)
            
            score = (len(rel1_support) - len(rel1_contradict)) / (
                len(rel1_support) + len(rel1_contradict) + 1e-10
            )
            consistency_scores.append(score)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _measure_independence(self, relationships: List[CompressionUnit]) -> float:
        """
        Measure context independence of relationships
        Higher scores indicate more universal patterns
        """
        if not relationships:
            return 0.0
            
        independence_scores = []
        
        for rel in relationships:
            # Count unique contexts
            contexts = set()
            for edge in rel.incoming_edges:
                for other_rel in relationships:
                    if edge in other_rel.outgoing_edges:
                        contexts.add(other_rel.tensor_position.channel)
            
            # Calculate independence score
            score = len(contexts) / len(relationships)
            independence_scores.append(score)
        
        return np.mean(independence_scores)
    
    def _measure_stability(self, relationships: List[CompressionUnit]) -> float:
        """
        Measure stability of relationship patterns
        Tracks consistency across variations
        """
        if not relationships:
            return 0.0
            
        stability_scores = []
        
        for rel in relationships:
            # Compare with recent history
            if rel.stability_history:
                variations = np.std(rel.stability_history)
                stability = 1.0 / (1.0 + variations)
                stability_scores.append(stability)
            
            # Check edge stability
            edge_stability = []
            for edge, weight in rel.outgoing_edges.items():
                if edge in rel.incoming_edges:
                    diff = abs(weight - rel.incoming_edges[edge])
                    edge_stability.append(1.0 / (1.0 + diff))
            
            if edge_stability:
                stability_scores.append(np.mean(edge_stability))
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _measure_generative_power(self, relationships: List[CompressionUnit]) -> float:
        """
        Measure ability of relationships to generate valid new patterns
        Key indicator of rational framework emergence
        """
        if len(relationships) < 2:
            return 0.0
            
        valid_patterns = 0
        total_patterns = 0
        
        # Try combining pairs of relationships
        for rel1, rel2 in zip(relationships[:-1], relationships[1:]):
            merged = rel1.merge_with(rel2)
            total_patterns += 1
            
            # Check if merged pattern is valid
            if self._is_valid_pattern(merged, relationships):
                valid_patterns += 1
        
        return valid_patterns / (total_patterns + 1e-10)
    
    def _measure_symmetry_increase(self, relationships: List[CompressionUnit]) -> float:
        """Measure rate of increase in symmetry scores"""
        if not relationships:
            return 0.0
            
        symmetry_scores = [r.symmetry_metrics.symmetry_score for r in relationships]
        if len(symmetry_scores) < 2:
            return 0.0
            
        return (symmetry_scores[-1] - symmetry_scores[0]) / len(symmetry_scores)
    
    def _measure_symmetry_stability(self, relationships: List[CompressionUnit]) -> float:
        """Measure stability of symmetry patterns"""
        if not relationships:
            return 0.0
            
        stability_scores = [r.symmetry_metrics.symmetry_stability for r in relationships]
        return np.mean(stability_scores)
    
    def _measure_symmetry_coherence(self, relationships: List[CompressionUnit]) -> float:
        """Measure how well symmetries align across relationships"""
        if len(relationships) < 2:
            return 0.0
            
        coherence_scores = []
        
        for rel1, rel2 in zip(relationships[:-1], relationships[1:]):
            # Compare transformation groups
            group_overlap = (rel1.symmetry_metrics.transformation_group & 
                           rel2.symmetry_metrics.transformation_group)
            group_coherence = len(group_overlap) / (
                len(rel1.symmetry_metrics.transformation_group | 
                    rel2.symmetry_metrics.transformation_group) + 1e-10
            )
            
            # Compare invariant properties
            prop_overlap = (rel1.symmetry_metrics.invariant_properties & 
                          rel2.symmetry_metrics.invariant_properties)
            prop_coherence = len(prop_overlap) / (
                len(rel1.symmetry_metrics.invariant_properties | 
                    rel2.symmetry_metrics.invariant_properties) + 1e-10
            )
            
            coherence_scores.append((group_coherence + prop_coherence) / 2)
        
        return np.mean(coherence_scores)
    
    def _is_transition_point(self, metrics: TransitionMetrics, level: int) -> bool:
        """Determine if current metrics indicate a transition point"""
        # Calculate metric derivatives
        derivatives = self._calculate_metric_derivatives(level)
        
        # Check for significant changes
        significant_changes = [
            abs(d) > self.threshold for d in derivatives.values()
        ]
        
        # Check for transition patterns
        transition_patterns = [
            metrics.self_reference > self.threshold,
            metrics.independence > self.threshold,
            metrics.generative_power > self.threshold,
            metrics.symmetry_increase > 0.2,
            any(significant_changes)
        ]
        
        return sum(transition_patterns) >= 3
    
    def _is_valid_pattern(self, pattern: CompressionUnit, 
                         relationships: List[CompressionUnit]) -> bool:
        """Check if a merged pattern is valid"""
        # Pattern should have meaningful connections
        if not pattern.incoming_edges and not pattern.outgoing_edges:
            return False
            
        # Pattern should maintain some consistency with existing relationships
        pattern_tokens = pattern.tokens
        support_count = sum(1 for rel in relationships 
                          if len(rel.tokens & pattern_tokens) / len(pattern_tokens) > 0.5)
        
        return support_count >= self.min_pattern_support
    
    def _calculate_metric_derivatives(self, level: int) -> Dict[str, float]:
        """Calculate rate of change for key metrics"""
        if level == 0 or not self.metric_history[level - 1]:
            return {}
            
        curr_metrics = self.metric_history[level][-1]
        prev_metrics = self.metric_history[level - 1][-1]
        
        metrics_vector = curr_metrics.as_vector()
        prev_vector = prev_metrics.as_vector()
        
        # Calculate derivatives
        derivatives = (metrics_vector - prev_vector) / 1.0  # Assuming unit time step
        
        # Map back to named metrics
        metric_names = [
            'abstraction', 'self_reference', 'consistency', 'independence',
            'stability', 'generative_power', 'symmetry_increase', 
            'symmetry_stability', 'symmetry_coherence'
        ]
        
        return {name: derivative for name, derivative in zip(metric_names, derivatives)}
    
    def _update_history(self, metrics: TransitionMetrics, level: int):
        """Update metric history"""
        self.metric_history[level].append(metrics)
        
        # Keep only recent history based on window size
        if len(self.metric_history[level]) > self.window_size:
            self.metric_history[level].pop(0)
    
    def _characterize_transition(self,
                               relationships: List[CompressionUnit],
                               metrics: TransitionMetrics,
                               level: int) -> TransitionPoint:
        """Characterize detected transition point in detail"""
        # Determine transition type based on dominant metrics
        if metrics.symmetry_increase > max(metrics.self_reference, 
                                         metrics.generative_power):
            transition_type = "symmetry_emergence"
        elif metrics.self_reference > metrics.generative_power:
            transition_type = "meta_relationship"
        else:
            transition_type = "pattern_generalization"
        
        # Calculate transition strength
        strength = np.mean([
            metrics.symmetry_increase,
            metrics.self_reference,
            metrics.generative_power,
            metrics.stability
        ])
        
        # Identify precursors and effects
        precursors = [rel.symbol for rel in relationships 
                     if rel.tensor_position.level < level]
        effects = [rel.symbol for rel in relationships 
                  if rel.tensor_position.level == level]
        
        return TransitionPoint(
            level=level,
            type=transition_type,
            strength=strength,
            metrics=metrics,
            precursors=precursors,
            effects=effects
        )