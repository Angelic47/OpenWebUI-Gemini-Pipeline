# AGENTS.md - Coding Guidelines for Gemini Pipeline

This file provides guidelines for AI coding agents working on this repository.

## Project Overview

This is an OpenWebUI Pipeline plugin for Google Gemini API integration. The project uses a **dual architecture**:
- **Development**: Modular code in `src/` directory (maintainable)
- **Distribution**: Single compiled file `gemini.py` (deployment)

## Build/Compile Commands

```bash
# Compile modular src/ into single gemini.py for distribution
python compile.py

# Verify compilation success
python -m py_compile gemini.py
python -m py_compile gemini_compile.py

# Full verification (compares functionality)
python verify_refactoring.py
```

## Test Commands

```bash
# Run equivalence test
python test_equivalence.py

# Run comprehensive verification
python verify_refactoring.py

# Run single test file
python -m pytest test_equivalence.py -v

# Run with coverage
python -m pytest test_equivalence.py --cov=src --cov-report=term-missing
```

## Lint/Format Commands

```bash
# Code formatting (use Black with 100 char line length)
black src/ compile.py test_*.py verify_*.py --line-length 100

# Import sorting
isort src/ compile.py --profile black

# Type checking
mypy src/ --ignore-missing-imports

# Linting
flake8 src/ --max-line-length 100 --extend-ignore=E203,W503
```

## Code Style Guidelines

### Import Order
1. Standard library imports
2. Third-party imports (google, fastapi, pydantic, PIL, etc.)
3. Local imports (from .module import ...)

```python
# Good example
import os
import re
from typing import Any, Optional

from google import genai
from PIL import Image

from .config import Valves
from .utils import strip_prefix
```

### Type Annotations
- Use Python 3.9+ syntax: `list[str]`, `dict[str, Any]` instead of `List`, `Dict`
- Use `|` for unions: `str | None` instead of `Optional[str]`
- Always annotate function parameters and return types

```python
def process_image(
    self, 
    image_data: str, 
    stats_list: list[dict[str, Any]] | None = None
) -> str:
    ...
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `ImageProcessor`, `ModelManager`)
- **Functions/Methods**: snake_case (e.g., `optimize_image`, `get_client`)
- **Constants**: UPPER_CASE_WITH_UNDERSCORES (e.g., `ASPECT_RATIO_OPTIONS`)
- **Private methods**: leading underscore (e.g., `_validate_api_key`)
- **Instance variables**: snake_case (e.g., `self.valves`, `self.log`)

### String Formatting
- Use double quotes for strings: `"string"` not `'string'`
- Use f-strings for formatting: `f"value: {variable}"`
- Multi-line strings use triple double quotes

### Documentation
- All modules must have a module-level docstring
- All classes must have a class-level docstring
- All public methods must have docstrings with Args/Returns sections
- Use Google-style docstrings

```python
def method_name(self, param: str) -> dict[str, Any]:
    """
    Brief description of what this method does.
    
    Args:
        param: Description of the parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this error occurs
    """
```

### Error Handling
- Use specific exception types (ValueError, TypeError, etc.)
- Log exceptions with `self.log.exception()` for full traceback
- Log warnings with `self.log.warning()`
- Log debug info with `self.log.debug()`

```python
try:
    result = await self._retry_with_backoff(get_response)
except ServerError as e:
    self.log.warning(f"Temporary error: {e}")
    raise
except Exception as e:
    self.log.exception(f"Unexpected error: {e}")
    return f"Error: {e}"
```

### Class Design
- Single Responsibility Principle: Each class does one thing
- Use dependency injection (pass valves/config to __init__)
- Initialize logger in __init__: `self.log = logging.getLogger("google_ai.pipe")`

### CRITICAL: Dynamic Configuration Access
**Never cache `valves` reference in helper classes!** OpenWebUI can change configuration at runtime.

✅ **Correct approach**: Pass `pipe` instance and access via `self.pipe.valves`
```python
class ImageProcessor:
    def __init__(self, pipe):
        self.pipe = pipe  # Store pipe reference
        self.log = logging.getLogger("google_ai.pipe")
    
    def optimize_image(self, image_data):
        # Always access current values
        if not self.pipe.valves.IMAGE_ENABLE_OPTIMIZATION:
            return image_data
        max_size = self.pipe.valves.IMAGE_MAX_SIZE_MB
        # ...
```

❌ **Wrong approach**: Caching valves reference
```python
class ImageProcessor:
    def __init__(self, valves):
        self.valves = valves  # DON'T DO THIS - won't see updates!
```

**Affected classes**: `ImageProcessor`, `ModelManager`, `ContentPreparator`, `GenerationConfigurator`

### Code Organization in src/
- One class per file (mostly)
- Helper functions in `utils.py`
- Constants in `constants.py`
- Main `Pipe` class in `pipe.py`

## Development Workflow

1. **Make changes** in `src/` directory files
2. **Run compile.py** to generate `gemini_compile.py`
3. **Run verify_refactoring.py** to ensure equivalence
4. **Test** the compiled output
5. **Copy** `gemini_compile.py` to `gemini.py` for final distribution

## Critical Rules

1. NEVER edit `gemini.py` directly - always edit files in `src/`
2. After any `src/` changes, you MUST run `compile.py`
3. Maintain backward compatibility with existing OpenWebUI installations
4. All new features must work in both modular and compiled forms
5. Don't break the existing API - the `Pipe` class interface is the contract

## Dependencies

Key dependencies (don't change versions without testing):
- `google-genai` - Google Gemini API client
- `pydantic` - Data validation
- `Pillow` - Image processing
- `fastapi` - Web framework (for types)
- `cryptography` - Encryption utilities

## Debugging Tips

- Use `self.log.debug()` liberally for troubleshooting
- Check logs in OpenWebUI for pipeline-specific messages
- Test with `verify_refactoring.py` after any structural changes
- Both files should be syntax-valid: `python -m py_compile <file>`

---

## Current Development Status (2026-02-11)

### 🚧 In Progress: Incremental Cache System

**Feature**: Smart cache rebuilding based on token count threshold

**Problem**: When conversation history grows, token costs explode. Need to:
1. Cache conversation history incrementally
2. Rebuild cache when non-cached content exceeds threshold (1x system prompt)
3. Handle user edits (signature mismatch detection)

**Architecture** (Two-Layer Caching):
```
┌─────────────────────────────────────┐
│  Layer 1: System Prompt Cache       │  ← Existing (cache.py)
│  - Caches system_instruction        │
│  - Unchanged logic                  │
├─────────────────────────────────────┤
│  Layer 2: Stage Cache (NEW)         │  ← Being implemented
│  - Caches conversation messages     │
│  - Named: "OpenWebUI Lumina Stage Cache"
│  - Contains: chat_id, cutoff_msg_id, signature
└─────────────────────────────────────┘
```

**Components Created/Modified**:
- ✅ `src/token_counter.py` - Token counting for cache decisions
- ✅ `src/utils.py` - Added `get_model_id_for_cache()` helper
- 🔄 `src/stage_cache.py` - NEW: Stage cache manager (next to implement)
- ⏳ `src/pipe.py` - Integration logic (after stage_cache.py)

**Design Document**: `INCREMENTAL_CACHE_MINDMAP.md`

### Next Steps (When We Return)

1. Create `src/stage_cache.py` with `StageCacheManager` class
   - `find_cache()` - lookup by chat_id and model_id
   - `create_cache()` - create new stage cache with signature
   - `delete_cache()` - remove old cache for chat_id
   - `compute_signature()` - content hash for mismatch detection

2. Integrate into `src/pipe.py`
   - Hook into generation flow
   - Check stage cache before generation
   - Token counting decision logic
   - Handle signature mismatch (force rebuild)

3. Test Scenarios:
   - Normal flow: cache found, signature matches
   - Token threshold: non-cached > 1x system prompt
   - Edit scenario: signature mismatch → force rebuild
   - Multi-modal: images in conversation history

**Key Design Decisions**:
- Signature stored in `display_name` (no extra API calls)
- One active cache per chat_id (delete old when creating new)
- Cutoff = last message id (keep incremental part minimal)
- Token counting only when no cache found (reduce API calls)
