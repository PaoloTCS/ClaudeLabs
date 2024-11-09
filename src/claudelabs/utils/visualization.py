# src/claudelabs/utils/visualization.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..core.data_structures import CompressionUnit, TransitionMetrics
from ..knowledge.intention import IntentionSpace, IntentState
from ..knowledge.curiosity import CuriosityCrawler, ExplorationResult


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    figsize: Tuple[int, int] = (12, 8)
    node_size: int = 1000
    font_size: int = 10
    edge_width: float = 1.5
    colormap: str = 'viridis'
    edge_colormap: str = 'coolwarm'
    dpi: int = 100
    
    # Style configurations
    background_color: str = 'white'
    grid_color: str = '#eeeeee'
    text_color: str = 'black'
    highlight_color: str = 'red'
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'figsize': self.figsize,
            'node_size': self.node_size,
            'font_size': self.font_size,
            'edge_width': self.edge_width,
            'dpi': self.dpi,
            'facecolor': self.background_color
        }


class KnowledgeVisualizer:
    """Visualization tools for knowledge structures and transitions"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.current_figure = None
        
    def plot_compression_graph(self, 
                             compression_units: List[CompressionUnit],
                             level: int,
                             show_symmetries: bool = False):
        """Visualize compression graph at specific level"""
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for unit in compression_units:
            if unit.tensor_position.level == level:
                G.add_node(unit.symbol,
                          compression_type=unit.compression_type,
                          symmetry=unit.symmetry_metrics.symmetry_score)
                
        # Add edges
        for u1, u2 in nx.combinations(compression_units, 2):
            if u1.tensor_position.level == level and u2.tensor_position.level == level:
                if self._has_relationship(u1, u2):
                    G.add_edge(u1.symbol, u2.symbol)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = [G.nodes[n]['symmetry'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, 
                             node_color=node_colors,
                             node_size=self.config.node_size,
                             cmap=plt.cm.get_cmap(self.config.colormap))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=self.config.edge_width)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=self.config.font_size)
        
        if show_symmetries:
            self._add_symmetry_indicators(G, pos)
            
        plt.title(f'Compression Graph - Level {level}')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(self.config.colormap)),
                    label='Symmetry Score')
        
        self.current_figure = plt.gcf()
        plt.show()
        
    def plot_transition_history(self,
                              transition_metrics: Dict[int, TransitionMetrics],
                              highlight_transitions: Optional[List[int]] = None):
        """Visualize transition metric evolution"""
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        levels = sorted(transition_metrics.keys())
        metrics_names = ['abstraction_level', 'self_reference', 
                        'symmetry_increase', 'stability']
        
        for metric in metrics_names:
            values = [getattr(transition_metrics[l], metric) for l in levels]
            plt.plot(levels, values, 'o-', label=metric)
            
        if highlight_transitions:
            for level in highlight_transitions:
                plt.axvline(x=level, color=self.config.highlight_color, 
                          linestyle='--', alpha=0.5)
            
        plt.xlabel('Compression Level')
        plt.ylabel('Metric Value')
        plt.title('Transition Metrics Evolution')
        plt.legend()
        plt.grid(True, color=self.config.grid_color)
        
        self.current_figure = plt.gcf()
        plt.show()
        
    def plot_intention_space(self,
                           intention_space: IntentionSpace,
                           highlight_path: Optional[List[str]] = None,
                           show_clusters: bool = True):
        """Visualize intention space with optional path highlight"""
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        G = intention_space.transitions
        pos = nx.spring_layout(G)
        
        # Draw basic graph
        node_colors = self._get_node_colors(intention_space, show_clusters)
        nx.draw_networkx_nodes(G, pos, 
                             node_color=node_colors,
                             node_size=self.config.node_size)
        
        # Draw edges
        if highlight_path:
            edge_colors = []
            edge_weights = []
            
            for u, v in G.edges():
                if highlight_path and u in highlight_path and v in highlight_path:
                    edge_colors.append(self.config.highlight_color)
                    edge_weights.append(2.0 * self.config.edge_width)
                else:
                    edge_colors.append('gray')
                    edge_weights.append(self.config.edge_width)
                    
            nx.draw_networkx_edges(G, pos,
                                 edge_color=edge_colors,
                                 width=edge_weights)
        else:
            nx.draw_networkx_edges(G, pos, width=self.config.edge_width)
            
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=self.config.font_size)
        
        plt.title('Intention Space')
        
        self.current_figure = plt.gcf()
        plt.show()
        
    def plot_exploration_results(self,
                               crawler: CuriosityCrawler,
                               show_metrics: bool = True):
        """Visualize exploration results and metrics"""
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        if show_metrics:
            gs = plt.GridSpec(2, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
        else:
            ax1 = fig.add_subplot(111)
        
        # Plot exploration path
        self._plot_exploration_path(crawler.exploration_history, ax1)
        
        if show_metrics:
            # Plot discovery metrics
            self._plot_discovery_metrics(crawler.exploration_history, ax2)
            
            # Plot coverage metrics
            self._plot_coverage_metrics(crawler, ax3)
        
        plt.tight_layout()
        self.current_figure = fig
        plt.show()
        
    def save_current_figure(self, filename: str):
        """Save current figure to file"""
        if self.current_figure:
            self.current_figure.savefig(filename, 
                                      dpi=self.config.dpi,
                                      bbox_inches='tight',
                                      facecolor=self.config.background_color)
        
    def _has_relationship(self, u1: CompressionUnit, u2: CompressionUnit) -> bool:
        """Check if two compression units have a relationship"""
        return bool(u1.tokens & u2.tokens)
        
    def _add_symmetry_indicators(self, G: nx.Graph, pos: Dict):
        """Add visual indicators for symmetry relationships"""
        for node, data in G.nodes(data=True):
            if data['symmetry'] > 0.7:  # High symmetry threshold
                plt.plot([pos[node][0]], [pos[node][1]], '*',
                        color=self.config.highlight_color,
                        markersize=20, alpha=0.5)
                
    def _get_node_colors(self, 
                        intention_space: IntentionSpace, 
                        show_clusters: bool) -> List:
        """Get node colors based on clustering or metrics"""
        if show_clusters and intention_space.state_clusters:
            # Color by cluster
            colors = []
            for node in intention_space.transitions.nodes():
                cluster_id = intention_space._get_cluster_id(node)
                colors.append(cluster_id if cluster_id is not None else -1)
        else:
            # Color by stability
            colors = [intention_space.stability_scores.get(node, 0.0)
                     for node in intention_space.transitions.nodes()]
            
        return colors
        
    def _plot_exploration_path(self, 
                             history: List[ExplorationResult],
                             ax: plt.Axes):
        """Plot exploration path with discoveries"""
        path = [result.state_id for result in history]
        novelty = [result.novelty_score for result in history]
        
        ax.plot(range(len(path)), novelty, 'b-', label='Novelty')
        ax.scatter(range(len(path)), novelty, c=novelty, 
                  cmap=plt.cm.get_cmap(self.config.colormap))
        
        ax.set_xlabel('Exploration Step')
        ax.set_ylabel('Novelty Score')
        ax.set_title('Exploration Path')
        ax.grid(True, color=self.config.grid_color)
        ax.legend()
        
    def _plot_discovery_metrics(self,
                              history: List[ExplorationResult],
                              ax: plt.Axes):
        """Plot discovery metrics over time"""
        steps = range(len(history))
        info_gain = [result.information_gain for result in history]
        
        ax.plot(steps, info_gain, 'g-', label='Information Gain')
        ax.fill_between(steps, 0, info_gain, alpha=0.3)
        
        ax.set_xlabel('Exploration Step')
        ax.set_ylabel('Information Gain')
        ax.set_title('Discovery Metrics')
        ax.grid(True, color=self.config.grid_color)
        ax.legend()
        
    def _plot_coverage_metrics(self,
                             crawler: CuriosityCrawler,
                             ax: plt.Axes):
        """Plot exploration coverage metrics"""
        metrics = crawler.get_exploration_metrics()
        
        # Create bar chart
        labels = ['States', 'Transitions', 'Coverage']
        values = [
            metrics['total_states_discovered'],
            metrics['total_transitions_discovered'],
            metrics['exploration_coverage']
        ]
        
        ax.bar(labels, values, color=plt.cm.get_cmap(self.config.colormap)(np.linspace(0, 1, 3)))
        ax.set_title('Exploration Coverage')
        ax.grid(True, color=self.config.grid_color)


class AnimationBuilder:
    """Helper class for creating animations of exploration and learning"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.frames = []
        
    def add_frame(self, fig: plt.Figure):
        """Add a figure as a frame to the animation"""
        self.frames.append(self._convert_figure_to_frame(fig))
        
    def save_animation(self, filename: str, fps: int = 2):
        """Save frames as an animation"""
        try:
            import imageio
            imageio.mimsave(filename, self.frames, fps=fps)
        except ImportError:
            print("imageio package required for saving animations")
            
    def _convert_figure_to_frame(self, fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to image array"""
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        return frame