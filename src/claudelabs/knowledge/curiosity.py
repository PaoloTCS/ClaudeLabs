# src/claudelabs/knowledge/curiosity.py

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import networkx as nx

from .intention import IntentionSpace, IntentState

@dataclass
class ExplorationResult:
    """Results from an exploration step"""
    state_id: str
    discovery_type: str
    novelty_score: float
    information_gain: float
    path_taken: List[str]
    properties_gained: Set[str]
    constraints_satisfied: Set[str]
    
    def as_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'state_id': self.state_id,
            'discovery_type': self.discovery_type,
            'novelty_score': self.novelty_score,
            'information_gain': self.information_gain,
            'path_taken': self.path_taken,
            'properties_gained': list(self.properties_gained),
            'constraints_satisfied': list(self.constraints_satisfied)
        }

class CuriosityCrawler:
    """Autonomous explorer with intent formation capabilities"""
    
    def __init__(self, intent_space: IntentionSpace):
        self.intent_space = intent_space
        self.current_state_id = None
        self.goal_state_id = None
        self.exploration_history = []
        self.discovered_states = set()
        self.discovered_transitions = set()
        self.exploration_weights = {
            'novelty': 0.4,
            'info_gain': 0.3,
            'stability': 0.2,
            'goal_alignment': 0.1
        }
        self.exploration_params = {
            'min_novelty': 0.2,
            'min_stability': 0.3,
            'max_steps_without_discovery': 5
        }
        
    def set_exploration_goal(self, state_id: str):
        """Set goal state for directed exploration"""
        if state_id in self.intent_space.states:
            self.goal_state_id = state_id
            
    def explore(self, steps: int) -> List[ExplorationResult]:
        """Conduct exploration steps, returning discoveries"""
        discoveries = []
        steps_without_discovery = 0
        
        for _ in range(steps):
            if steps_without_discovery >= self.exploration_params['max_steps_without_discovery']:
                # Reset to random unexplored state to break out of local minimum
                self._reset_exploration()
                steps_without_discovery = 0
                
            if self.goal_state_id:
                next_state = self._directed_step()
            else:
                next_state = self._curiosity_step()
                
            if next_state:
                discovery = self._document_discovery(next_state)
                if discovery.novelty_score > self.exploration_params['min_novelty']:
                    discoveries.append(discovery)
                    steps_without_discovery = 0
                else:
                    steps_without_discovery += 1
                    
                self.current_state_id = next_state
                
        return discoveries
    
    def get_exploration_metrics(self) -> Dict[str, float]:
        """Get metrics about the exploration process"""
        if not self.exploration_history:
            return {}
            
        discoveries = [d for d in self.exploration_history 
                      if d.novelty_score > self.exploration_params['min_novelty']]
        
        return {
            'total_states_discovered': len(self.discovered_states),
            'total_transitions_discovered': len(self.discovered_transitions),
            'average_novelty': np.mean([d.novelty_score for d in discoveries]) 
                             if discoveries else 0.0,
            'average_information_gain': np.mean([d.information_gain for d in discoveries])
                                      if discoveries else 0.0,
            'exploration_coverage': len(self.discovered_states) / 
                                  len(self.intent_space.states)
        }
    
    def _directed_step(self) -> Optional[str]:
        """Take step toward goal using necessary paths"""
        if not self.current_state_id or not self.goal_state_id:
            return None
            
        path = self.intent_space.find_necessary_path(
            self.current_state_id, 
            self.goal_state_id
        )
        
        if len(path) >= 2:
            # Consider multiple candidates near the path
            candidates = self._get_path_candidates(path[1])
            return self._select_best_candidate(candidates)
            
        return None
    
    def _curiosity_step(self) -> Optional[str]:
        """Take step based on novelty and potential information gain"""
        if not self.current_state_id:
            # Start with random unexplored state
            unexplored = set(self.intent_space.states.keys()) - self.discovered_states
            if unexplored:
                return np.random.choice(list(unexplored))
            return None
            
        # Get possible next states
        candidates = self._get_exploration_candidates()
        if not candidates:
            return None
            
        return self._select_best_candidate(candidates)
    
    def _get_exploration_candidates(self) -> List[str]:
        """Get candidate states for exploration"""
        candidates = set()
        
        # Direct neighbors
        neighbors = list(self.intent_space.transitions.neighbors(self.current_state_id))
        candidates.update(neighbors)
        
        # Similar states
        similar_states = self.intent_space.get_similar_states(
            self.current_state_id, 
            threshold=0.5
        )
        candidates.update(similar_states)
        
        # States in same cluster
        cluster_id = self.intent_space._get_cluster_id(self.current_state_id)
        if cluster_id is not None:
            candidates.update(self.intent_space.state_clusters[cluster_id])
        
        return list(candidates - {self.current_state_id})
    
    def _get_path_candidates(self, next_state: str) -> List[str]:
        """Get candidate states near the planned path"""
        candidates = {next_state}
        
        # Include similar states
        similar_states = self.intent_space.get_similar_states(
            next_state,
            threshold=0.7
        )
        candidates.update(similar_states)
        
        # Include states from same cluster
        cluster_id = self.intent_space._get_cluster_id(next_state)
        if cluster_id is not None:
            candidates.update(self.intent_space.state_clusters[cluster_id])
        
        return list(candidates)
    
    def _select_best_candidate(self, candidates: List[str]) -> Optional[str]:
        """Select best candidate based on multiple criteria"""
        if not candidates:
            return None
            
        scores = []
        current_state = self.intent_space.states[self.current_state_id]
        
        for state_id in candidates:
            state = self.intent_space.states[state_id]
            score = self._compute_state_score(state, current_state)
            scores.append((score, state_id))
            
        if scores:
            return max(scores, key=lambda x: x[0])[1]
        return None
    
    def _compute_state_score(self, 
                           state: IntentState,
                           current_state: IntentState) -> float:
        """Compute overall score for a candidate state"""
        # Novelty component
        novelty = (1.0 if state.id not in self.discovered_states 
                  else 0.2)
        
        # Information gain component
        new_properties = len(state.properties - current_state.properties)
        new_constraints = len(state.constraints - current_state.constraints)
        total_elements = len(state.properties) + len(state.constraints)
        info_gain = (new_properties + new_constraints) / (total_elements + 1e-10)
        
        # Stability component
        stability = self.intent_space.stability_scores.get(state.id, 0.0)
        
        # Goal alignment component
        goal_alignment = 0.0
        if self.goal_state_id:
            goal_state = self.intent_space.states[self.goal_state_id]
            goal_alignment = state.compatibility_score(goal_state)
        
        # Combine components
        return (
            self.exploration_weights['novelty'] * novelty +
            self.exploration_weights['info_gain'] * info_gain +
            self.exploration_weights['stability'] * stability +
            self.exploration_weights['goal_alignment'] * goal_alignment
        )
    
    def _document_discovery(self, state_id: str) -> ExplorationResult:
        """Document state discovery and update history"""
        state = self.intent_space.states[state_id]
        current_state = (self.intent_space.states[self.current_state_id] 
                        if self.current_state_id else None)
        
        # Calculate properties gained
        properties_gained = (set() if current_state is None 
                           else state.properties - current_state.properties)
        
        # Calculate constraints satisfied
        constraints_satisfied = (set() if current_state is None
                               else state.constraints & current_state.properties)
        
        # Determine discovery type
        discovery_type = self._determine_discovery_type(state, current_state)
        
        # Calculate novelty and information gain
        novelty_score = 1.0 if state_id not in self.discovered_states else 0.2
        information_gain = (len(properties_gained) + len(constraints_satisfied)) / (
            len(state.properties) + len(state.constraints)
        ) if state.properties or state.constraints else 0.0
        
        # Record discovery
        discovery = ExplorationResult(
            state_id=state_id,
            discovery_type=discovery_type,
            novelty_score=novelty_score,
            information_gain=information_gain,
            path_taken=self._get_current_path(),
            properties_gained=properties_gained,
            constraints_satisfied=constraints_satisfied
        )
        
        self.discovered_states.add(state_id)
        if current_state:
            self.discovered_transitions.add((current_state.id, state_id))
        self.exploration_history.append(discovery)
        
        return discovery
    
    def _determine_discovery_type(self, 
                                state: IntentState,
                                current_state: Optional[IntentState]) -> str:
        """Determine the type of discovery made"""
        if not current_state:
            return "initial_state"
            
        if state.id not in self.discovered_states:
            return "new_state"
            
        if len(state.properties - current_state.properties) > 0:
            return "property_expansion"
            
        if len(state.constraints & current_state.properties) > 0:
            return "constraint_satisfaction"
            
        return "revisit"
    
    def _get_current_path(self) -> List[str]:
        """Get the path taken to current state"""
        path = []
        for discovery in self.exploration_history:
            path.append(discovery.state_id)
        return path
    
    def _reset_exploration(self):
        """Reset exploration to break out of local minimum"""
        unexplored = set(self.intent_space.states.keys()) - self.discovered_states
        if unexplored:
            self.current_state_id = np.random.choice(list(unexplored))
        else:
            # If all states explored, choose random state
            self.current_state_id = np.random.choice(list(self.intent_space.states.keys()))