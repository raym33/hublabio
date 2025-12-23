# Contributing to HubLab IO

Thank you for your interest in contributing to HubLab IO!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hublabio.git
   ```
3. Set up development environment:
   ```bash
   make setup
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

## Development Workflow

### Code Style

We use `rustfmt` and `clippy`:

```bash
# Format code
make fmt

# Lint code
make lint

# Check compilation
make check
```

### Testing

```bash
# Run all tests
make test

# Run specific component tests
make test-kernel
make test-runtime
make test-sdk
```

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(kernel): add memory-mapped AI model support
fix(scheduler): correct priority inversion bug
docs(sdk): add AI client examples
```

### Pull Requests

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review

## Architecture Guidelines

### Kernel Code
- Keep it minimal - move functionality to userspace when possible
- No allocations in interrupt handlers
- Use `#![no_std]` only

### Runtime Code
- Support both `std` and `no_std` when possible
- Minimize dependencies
- Document public APIs

### SDK Code
- Provide ergonomic APIs
- Include examples
- Maintain backwards compatibility

## Areas for Contribution

### High Priority
- Device driver implementations
- Filesystem implementations
- Network stack improvements
- AI model optimizations

### Medium Priority
- Documentation improvements
- Test coverage
- Performance optimizations
- New shell commands

### Low Priority
- GUI widgets
- Additional themes
- Localization

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Assume good intentions

## Questions?

- Open an issue for bugs
- Use discussions for questions
- Join our community chat

Thank you for contributing!
