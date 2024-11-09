# src/claudelabs/knowledge/intention.py

from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

@dataclass
class IntentState:
    """Represents a state in the intention space"""
    id: str
    properties: Set[str]     # Observable properties
    constraints: Set[str]    # Must-maintain conditions
    proof_chain: List[str]  # Sequence of validations
    satisfaction_level: float = 0.0  # Degree of constraint satisfaction
    
    def validates(self, target_state: 'IntentState') -> bool:
        """Check if this state validates progression to target"""
        return all(p in self.properties for p in target_state.constraints)
    
    def compatibility_score(self, other: 'IntentState') -> float:
        """Calculate compatibility with another state"""
        shared_properties = len(self.properties & other.properties)
        total_properties = len(self.properties | other.properties)
        shared_constraints = len(self.constraints & other.constraints)
        total_constraints = len(self.constraints | other.constraints)
        
        property_score = shared_properties / total_properties if total_properties > 0 else 0
        constraint_score = shared_constraints / total_constraints if total_constraints > 0 else 0
        
        return 0.7 * property_score + 0.3 * constraint_score
    
    def merge_with(self, other: 'IntentState') -> 'IntentState':
        """Create a new state by merging with another state"""
        return IntentState(
            id=f"{self.id}+{other.id}",
            properties=self.properties | other.properties,
            constraints=self.constraints | other.constraints,
            proof_chain=self.proof_chain + other.proof_chain,
            satisfaction_level=(self.satisfaction_level + other.satisfaction_level) / 2
        )

class IntentionSpace:
    """Models the space of possible intentions and their transitions"""
    
    def __init__(self):
        self.states = {}          # id -> IntentState
        self.transitions = nx.DiGraph()
        self.necessity_paths = {} # (start, end) -> [necessary states]
        self.state_clusters = {}  # cluster_id -> set of state ids
        self.stability_scores = {}  # state_id -> stability score
        
    def add_state(self, state: IntentState):
        """Add a state to the intention space"""
        self.states[state.id] = state
        self.transitions.add_node(state.id)
        self._update_clusters()
        self._calculate_stability(state.id)
    
    def add_transition(self, from_id: str, to_id: str, 
                      proof: List[str], weight: float = 1.0):
        """Add a validated transition between states"""
        if from_id not in self.states or to_id not in self.states:
            raise ValueError("Both states must exist in the space")
            
        if self.states[from_id].validates(self.states[to_id]):
            self.transitions.add_edge(
                from_id, to_id, 
                proof=proof, 
                weight=weight
            )
            self._update_clusters()
            
    def find_necessary_path(self, start_id: str, end_id: str) -> List[str]:
        """Find path where each intermediate state is necessary"""
        key = (start_id, end_id)
        if key in self.necessity_paths:
            return self.necessity_paths[key]
            
        paths = list(nx.all_simple_paths(self.transitions, start_id, end_id))
        necessary_paths = []
        
        for path in paths:
            if self._is_necessary_path(path):
                necessary_paths.append(path)
                
        if necessary_paths:
            shortest = min(necessary_paths, key=len)
            self.necessity_paths[key] = shortest
            return shortest
        return []
    
    def get_similar_states(self, state_id: str, threshold: float = 0.7) -> List[str]:
        """Find states similar to the given state"""
        if state_id not in self.states:
            return []
            
        state = self.states[state_id]
        similarities = []
        
        for other_id, other_state in self.states.items():
            if other_id != state_id:
                score = state.compatibility_score(other_state)
                if score >= threshold:
                    similarities.append((score, other_id))
                    
        return [s[1] for s in sorted(similarities, reverse=True)]
    
    def get_state_metrics(self, state_id: str) -> Dict[str, float]:
        """Get comprehensive metrics for a state"""
        if state_id not in self.states:
            return {}
            
        state = self.states[state_id]
        
        # Calculate centrality in transition graph
        centrality = nx.pagerank(self.transitions).get(state_id, 0.0)
        
        # Calculate cluster cohesion
        cluster_id = self._get_cluster_id(state_id)
        cluster_cohesion = self._calculate_cluster_cohesion(cluster_id)
        
        return {
            'satisfaction_level': state.satisfaction_level,
            'centrality': centrality,
            'stability': self.stability_scores.get(state_id, 0.0),
            'cluster_cohesion': cluster_cohesion,
            'outgoing_transitions': len(list(self.transitions.successors(state_id))),
            'incoming_transitions': len(list(self.transitions.predecessors(state_id)))
        }
    
    def _is_necessary_path(self, path: List[str]) -> bool:
        """Check if each state in path is necessary for progression"""
        if len(path) <= 2:
            return True
            
        for i in range(1, len(path)-1):
            prev_state = self.states[path[i-1]]
            curr_state = self.states[path[i]]
            next_state = self.states[path[i+1]]
            
            # Properties provided uniquely by current state
            unique_properties = curr_state.properties - prev_state.properties
            
            # Check if any unique properties are required by next state
            if not (unique_properties & next_state.constraints):
                return False
        return True
    
    def _update_clusters(self):
        """Update state clusters based on current transitions"""
        # Use community detection to identify clusters
        try:
            import community  # python-louvain
            partition = community.best_partition(
                self.transitions.to_undirected()
            )
            
            # Group states by cluster
            clusters = {}
            for state_id, cluster_id in partition.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = set()
                clusters[cluster_id].add(state_id)
            
            self.state_clusters = clusters
            
        except ImportError:
            # Fallback to simple clustering based on connectivity
            connected = nx.connected_components(self.transitions.to_undirected())
            self.state_clusters = {i: cluster for i, cluster in enumerate(connected)}
    
    def _calculate_stability(self, state_id: str):
        """Calculate stability score for a state"""
        state = self.states[state_id]
        
        # Consider multiple factors for stability
        factors = []
        
        # 1. Consistency of outgoing transitions
        out_neighbors = list(self.transitions.successors(state_id))
        if out_neighbors:
            consistencies = []
            for neighbor_id in out_neighbors:
                neighbor = self.states[neighbor_id]
                consistencies.append(state.compatibility_score(neighbor))
            factors.append(np.mean(consistencies))
        
        # 2. Satisfaction of constraints
        factors.append(state.satisfaction_level)
        
        # 3. Cluster cohesion
        cluster_id = self._get_cluster_id(state_id)
        if cluster_id is not None:
            factors.append(self._calculate_cluster_cohesion(cluster_id))
        
        # Combine factors
        self.stability_scores[state_id] = np.mean(factors) if factors else 0.0
    
    def _get_cluster_id(self, state_id: str) -> Optional[int]:
        """Get cluster ID for a state"""
        for cluster_id, states in self.state_clusters.items():
            if state_id in states:
                return cluster_id
        return None
    
    def _calculate_cluster_cohesion(self, cluster_id: int) -> float:
        """Calculate cohesion of a cluster"""
        if cluster_id not in self.state_clusters:
            return 0.0
            
        states = self.state_clusters[cluster_id]
        if len(states) < 2:
            return 1.0
            
        # Calculate average compatibility between states in cluster
        scores = []
        for s1 in states:
            for s2 in states:
                if s1 < s2:  # Avoid duplicate comparisons
                    scores.append(
                        self.states[s1].compatibility_score(self.states[s2])
                    )
                    
        return np.mean(scores) if scores else 0.0