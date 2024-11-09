

# ClaudeLabs

A comprehensive framework for knowledge representation, analysis, and exploration, featuring tensor-based compression, symmetry detection, and curiosity-driven learning.

## Features

- **Knowledge Representation**
  - Tensor-based compression
  - Hierarchical knowledge structures
  - Symmetry detection and analysis

- **Analysis Tools**
  - Network analysis integration
  - Transition detection
  - Pattern recognition

- **Exploration Capabilities**
  - Intention space modeling
  - Curiosity-driven exploration
  - Automated discovery

## Installation

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/PaoloTCS/claudelabs.git
cd claudelabs
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### User Installation

```bash
pip install claudelabs
```

## Quick Start

Here's a simple example of using ClaudeLabs:

```python
from claudelabs.core.data_structures import CompressionUnit, TensorPosition
from claudelabs.analysis.network_analysis import NetworkAnalysisIntegrator

# Create sample compression units
unit = CompressionUnit(
    tokens={"data", "processing"},
    symbol="dp1",
    tensor_position=TensorPosition(level=1, position=0, channel=0),
    compression_type=CompressionType.SEQUENTIAL
)

# Initialize network analyzer
analyzer = NetworkAnalysisIntegrator(tensor_dims=(3, 2, 1))
graph = analyzer.build_level_graph(1, [unit])
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black src/claudelabs tests/

# Sort imports
isort src/claudelabs tests/
```

### Type Checking

```bash
mypy src/claudelabs
```

## Project Structure

```
claudelabs/
├── src/
│   └── claudelabs/
│       ├── core/          # Core data structures and operations
│       ├── analysis/      # Analysis tools and algorithms
│       ├── knowledge/     # Knowledge representation and learning
│       └── utils/         # Utility functions and helpers
├── tests/                 # Test suite
├── docs/                  # Documentation
├── requirements.txt       # Project dependencies
└── setup.py              # Package configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Paolo Pignatelli - Paolodim@gmail.com
Project Link: https://github.com/PaoloTCS/claudelabs