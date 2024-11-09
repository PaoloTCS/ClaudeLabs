# src/claudelabs/analysis/network_analysis.py

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import itertools
import community  # python-louvain package

from ..core.data_structures import CompressionUnit, TensorPosition, SymmetryMetrics

@dataclass
class NetworkAnalysisResult:
    """Results from network analysis across compression levels"""
    communities: Dict[int, List[Set[str]]]  # level -> list of communities
    centrality_scores: Dict[int, Dict[str, float]]  # level -> node -> score
    structural_roles: Dict[int, Dict[str, int]]  # level -> node -> role
    meta_patterns: List[Tuple[int, Set[str]]]  # (level, pattern) pairs
    
    def get_community_metrics(self, level: int) -> Dict[str, float]:
        """Calculate community-level metrics for a given level"""
        if level not in self.communities:
            return {}
            
        return {
            'num_communities': len(self.communities[level]),
            'avg_community_size': np.mean([len(c) for c in self.communities[level]]),
            'max_community_size': max(len(c) for c in self.communities[level]),
            'community_density': self._calculate_community_density(level)
        }
        
    def _calculate_community_density(self, level: int) -> float:
        """Calculate the density of community connections"""
        if level not in self.communities:
            return 0.0
            
        total_nodes = sum(len(c) for c in self.communities[level])
        total_internal_edges = sum(
            len(c) * (len(c) - 1) / 2 for c in self.communities[level]
        )
        max_possible_edges = total_nodes * (total_nodes - 1) / 2
        
        return total_internal_edges / max_possible_edges if max_possible_edges > 0 else 0.0

class NetworkAnalysisIntegrator:
    """Integrates network analysis with tensor compression framework"""
    
    def __init__(self, tensor_dims: Tuple[int, int, int]):
        self.L, self.P, self.C = tensor_dims  # tensor dimensions
        self.compression_graphs = {}  # level -> NetworkX graph
        self.analysis_results = {}  # level -> NetworkAnalysisResult
        
    def build_level_graph(self, 
                         level: int, 
                         compression_units: List[CompressionUnit]) -> nx.Graph:
        """Constructs relationship graph for a given compression level"""
        G = nx.Graph()
        
        # Add nodes with compression metadata
        for unit in compression_units:
            if unit.tensor_position.level == level:
                G.add_node(unit.symbol, 
                          tokens=unit.tokens,
                          tensor_pos=unit.tensor_position,
                          compression_type=unit.compression_type,
                          symmetry_metrics=unit.symmetry_metrics)
        
        # Add edges based on relationships
        for u1, u2 in itertools.combinations(compression_units, 2):
            if u1.tensor_position.level == level and u2.tensor_position.level == level:
                similarity = self._compute_relationship_similarity(u1, u2)
                if similarity > 0:
                    G.add_edge(u1.symbol, u2.symbol, weight=similarity)
        
        self.compression_graphs[level] = G
        return G
    
    def analyze_level(self, level: int) -> NetworkAnalysisResult:
        """Performs comprehensive network analysis for a compression level"""
        G = self.compression_graphs.get(level)
        if not G:
            raise ValueError(f"No graph found for level {level}")
        
        # Community detection
        communities = self._detect_communities(G)
        
        # Centrality analysis
        centrality = self._compute_centrality_measures(G)
        
        # Structural role analysis
        roles = self._identify_structural_roles(G)
        
        # Meta-pattern detection
        meta_patterns = self._detect_meta_patterns(G, communities, roles)
        
        result = NetworkAnalysisResult(
            communities={level: communities},
            centrality_scores={level: centrality},
            structural_roles={level: roles},
            meta_patterns=meta_patterns
        )
        
        self.analysis_results[level] = result
        return result
    
    def _detect_communities(self, G: nx.Graph) -> List[Set[str]]:
        """Detects communities using Louvain method"""
        partition = community.best_partition(G)
        
        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)
            
        return list(communities.values())
    
    def _compute_centrality_measures(self, G: nx.Graph) -> Dict[str, float]:
        """Computes multiple centrality measures and combines them"""
        # Eigenvector centrality
        eigen_cent = nx.eigenvector_centrality_numpy(G)
        
        # Betweenness centrality
        between_cent = nx.betweenness_centrality(G)
        
        # Combine measures with weights
        combined = {}
        for node in G.nodes():
            combined[node] = (
                0.7 * eigen_cent[node] + 
                0.3 * between_cent[node]
            )
            
        return combined
    
    def _identify_structural_roles(self, G: nx.Graph) -> Dict[str, int]:
        """Identifies structural roles using role-based analysis"""
        roles = {}
        
        for node in G.nodes():
            # Compute local features
            degree = G.degree(node)
            clustering = nx.clustering(G, node)
            neighbors = set(G.neighbors(node))
            
            # Assign roles based on features
            if degree > 0.7 * len(G):
                roles[node] = 0  # Hub
            elif clustering > 0.7:
                roles[node] = 1  # Clique member
            elif any(G.degree(n) > degree for n in neighbors):
                roles[node] = 2  # Bridge
            else:
                roles[node] = 3  # Peripheral
                
        return roles
    
    def _detect_meta_patterns(self,
                            G: nx.Graph,
                            communities: List[Set[str]],
                            roles: Dict[str, int]) -> List[Tuple[int, Set[str]]]:
        """Detects meta-patterns from community and role structure"""
        meta_patterns = []
        
        # Analyze inter-community patterns
        for comm1, comm2 in itertools.combinations(communities, 2):
            pattern = self._analyze_community_interaction(G, comm1, comm2)
            if pattern:
                meta_patterns.append((len(meta_patterns), pattern))
        
        # Analyze role-based patterns
        role_patterns = self._analyze_role_patterns(G, roles)
        meta_patterns.extend((len(meta_patterns) + i, p) 
                           for i, p in enumerate(role_patterns))
        
        return meta_patterns
    
    def _analyze_community_interaction(self,
                                    G: nx.Graph,
                                    comm1: Set[str],
                                    comm2: Set[str]) -> Optional[Set[str]]:
        """Analyze interaction pattern between communities"""
        edges = [(u, v) for u, v in G.edges() 
                if (u in comm1 and v in comm2) or (u in comm2 and v in comm1)]
        
        if not edges:
            return None
            
        # Find nodes involved in inter-community connections
        connected_nodes = {u for e in edges for u in e}
        return connected_nodes
    
    def _analyze_role_patterns(self,
                             G: nx.Graph,
                             roles: Dict[str, int]) -> List[Set[str]]:
        """Analyze patterns in role distributions"""
        patterns = []
        
        # Group nodes by role
        role_groups = {role: {n for n, r in roles.items() if r == role} 
                      for role in set(roles.values())}
        
        # Find connected role groups
        for role1, nodes1 in role_groups.items():
            for role2, nodes2 in role_groups.items():
                if role1 >= role2:
                    continue
                    
                connections = {(u, v) for u in nodes1 for v in nodes2 
                             if G.has_edge(u, v)}
                if connections:
                    pattern_nodes = {u for e in connections for u in e}
                    patterns.append(pattern_nodes)
        
        return patterns
    
    def _compute_relationship_similarity(self,
                                      u1: CompressionUnit,
                                      u2: CompressionUnit) -> float:
        """Computes similarity between compression units"""
        token_sim = len(u1.tokens & u2.tokens) / len(u1.tokens | u2.tokens)
        pos_sim = 1 - (abs(u1.tensor_position.position - 
                          u2.tensor_position.position) / self.P)
        sym_sim = (u1.symmetry_metrics.symmetry_score + 
                  u2.symmetry_metrics.symmetry_score) / 2
        
        return 0.5 * token_sim + 0.3 * pos_sim + 0.2 * sym_sim
    
    def get_network_metrics(self, level: int) -> Dict[str, float]:
        """Calculate comprehensive network metrics for a level"""
        G = self.compression_graphs.get(level)
        if not G:
            return {}
            
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'avg_shortest_path': nx.average_shortest_path_length(G)
            if nx.is_connected(G) else float('inf'),
        }
        
        if level in self.analysis_results:
            metrics.update(self.analysis_results[level].get_community_metrics(level))
            
        return metrics