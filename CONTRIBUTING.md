# Contributing to Dynamic Information Lattices

We welcome contributions to the Dynamic Information Lattices project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Installation

```bash
git clone https://github.com/your-username/dynamic-info-lattices.git
cd dynamic-info-lattices
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Style

We use Black for code formatting and flake8 for linting:

```bash
black dynamic_info_lattices/ examples/ tests/
flake8 dynamic_info_lattices/ examples/ tests/
```

## Types of Contributions

### Bug Reports

When filing a bug report, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages and stack traces

### Feature Requests

For feature requests, please provide:

- A clear description of the proposed feature
- Use cases and motivation
- Possible implementation approach
- Any relevant research papers or references

### Code Contributions

#### Pull Request Process

1. **Create a branch**: Use a descriptive name like `feature/new-entropy-component` or `fix/memory-leak`

2. **Make changes**: 
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit messages**: Use clear, descriptive commit messages:
   ```
   Add spectral entropy component for frequency analysis
   
   - Implement FFT-based spectral entropy estimation
   - Add unit tests for spectral entropy calculation
   - Update documentation with usage examples
   ```

4. **Submit PR**: 
   - Provide a clear description of changes
   - Reference any related issues
   - Include test results
   - Request review from maintainers

#### Code Guidelines

- **Documentation**: All public functions and classes should have docstrings
- **Type hints**: Use type hints for function parameters and return values
- **Error handling**: Include appropriate error handling and validation
- **Performance**: Consider computational efficiency, especially for core algorithms
- **Compatibility**: Ensure compatibility with supported Python and PyTorch versions

#### Testing Guidelines

- Write unit tests for new functionality
- Include integration tests for complex features
- Test edge cases and error conditions
- Ensure tests are deterministic and reproducible
- Add performance benchmarks for core algorithms

## Research Contributions

### Algorithm Improvements

If you're contributing algorithmic improvements:

- Provide theoretical justification
- Include experimental validation
- Compare against existing methods
- Document computational complexity
- Provide reproducible experiments

### New Datasets

For new dataset contributions:

- Ensure proper licensing and attribution
- Provide data preprocessing scripts
- Include dataset statistics and characteristics
- Add evaluation benchmarks
- Document data format and structure

### Baseline Methods

When adding baseline methods:

- Implement fair comparisons
- Use consistent evaluation protocols
- Provide proper citations
- Include hyperparameter tuning
- Document implementation details

## Documentation

### Code Documentation

- Use clear, descriptive docstrings
- Follow Google or NumPy docstring style
- Include parameter types and descriptions
- Provide usage examples
- Document any assumptions or limitations

### User Documentation

- Update README.md for new features
- Add examples to the examples/ directory
- Create tutorials for complex workflows
- Update API documentation
- Include performance benchmarks

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge contributions from others
- Maintain a professional tone

### Communication

- Use GitHub issues for bug reports and feature requests
- Use pull requests for code contributions
- Join our Discord for discussions (link in README)
- Follow up on your contributions

## Release Process

### Version Numbering

We follow semantic versioning (SemVer):
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] Release notes prepared
- [ ] Performance benchmarks run

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Academic papers (for significant contributions)
- Conference presentations

## Questions?

If you have questions about contributing:

- Check existing issues and documentation
- Ask in GitHub discussions
- Join our Discord community
- Contact the maintainers directly

Thank you for contributing to Dynamic Information Lattices!
