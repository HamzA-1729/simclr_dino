# Contributing to `simclr_dino`

Contributions, bug reports, and mathematical verification improvements are welcome.

## Development Setup

1. Clone the repository and sync the isolated `uv` environment:
   ```bash
   git clone https://github.com/HamzA-1729/simclr_dino.git
   cd simclr_dino
   uv sync
   ```
2. Run the automated test suite before submitting changes:
   ```bash
   uv run pytest -v
   ```

## Standards
- Keep implementations pure PyTorch without high-level SSL wrappers.
- Every new mathematical invariant, augmentation, or loss property must include a deterministic unit test in `tests/`.
