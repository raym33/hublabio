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

# Run in QEMU
make run
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
feat(shell): add GUI compositor
feat(voice): implement Piper TTS
feat(pkg): add package manager with repository support
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
- Support SMP (multi-core) designs

### Runtime Code
- Support both `std` and `no_std` when possible
- Minimize dependencies
- Document public APIs

### Shell Code
- Support all three modes (TUI, GUI, Voice)
- Use ANSI escape codes for TUI
- Keep widgets reusable

### SDK Code
- Provide ergonomic APIs
- Include examples
- Maintain backwards compatibility

### Package Manager Code
- Support offline operation when possible
- Handle dependency resolution correctly
- Use proper versioning (semver)

## Project Structure

```
hublabio/
├── kernel/          # Microkernel (Rust, no_std)
├── runtime/         # AI inference engine
├── shell/           # User interfaces
│   ├── tui/        # Terminal interface
│   ├── gui/        # Graphical interface
│   └── voice/      # Voice interface
├── apps/           # System applications
│   ├── core/       # Core apps (files, terminal)
│   └── system/     # System apps (monitor, settings)
├── pkg/            # Package manager
│   ├── manager/    # Package management
│   └── repo/       # Repository handling
├── sdk/            # Application SDK
├── hal/            # Hardware abstraction
├── services/       # System services
├── docs/           # Documentation
└── tools/          # Build tools
```

## Areas for Contribution

### High Priority
- Real hardware testing on Raspberry Pi
- MoE-R distributed experts implementation
- NPU/GPU acceleration
- Performance optimization

### Medium Priority
- Additional device drivers
- New shell themes
- Test coverage improvements
- Documentation translations

### Low Priority
- New system apps
- Additional GUI widgets
- Voice command extensions
- Localization

## Component-Specific Guidelines

### TUI Shell
- Use the `ansi` module for escape codes
- Implement `InputEvent` handling in mod.rs
- Support all themes (Material Dark, AMOLED, Light)

### GUI Compositor
- Use framebuffer for rendering
- Implement proper z-ordering for windows
- Handle touch and mouse input

### Voice Interface
- Use Whisper-compatible models for STT
- Use Piper-compatible models for TTS
- Implement wake word detection

### Package Manager
- Use semver for versions
- Support repository priorities
- Handle dependency conflicts gracefully

### Settings App
- Group settings by category
- Support different value types
- Persist changes to configuration

## Testing Guidelines

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Test implementation
    }
}
```

### Integration Tests
```bash
# Run integration tests
make test-integration
```

### Hardware Tests
Document your hardware testing:
- Device model
- RAM size
- Any peripherals
- Test results

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Assume good intentions

## Getting Help

- Open an issue for bugs
- Use discussions for questions
- Check existing documentation
- Review closed issues/PRs for context

## Recognition

Contributors are recognized in:
- CHANGELOG.md for each release
- GitHub contributors page
- Project README (major contributors)

Thank you for contributing to HubLab IO!
