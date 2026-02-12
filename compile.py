#!/usr/bin/env python3
"""
Compile script for Gemini Pipeline.
Combines all source files in src/ into a single gemini_compile.py file.
"""

import re
from pathlib import Path


# OpenWebUI required header (must be at the very top of the file)
OPENWEBUI_HEADER = '''"""
title: Google Gemini Pipeline
author: owndev, olivier-lacroix
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.10.0
required_open_webui_version: 0.6.26
license: Apache License 2.0
description: Highly optimized Google Gemini pipeline with advanced image generation capabilities, intelligent compression, and streamlined processing workflows.
features:
  - Optimized asynchronous API calls for maximum performance
  - Intelligent model caching with configurable TTL
  - Streamlined dynamic model specification with automatic prefix handling
  - Smart streaming response handling with safety checks
  - Advanced multimodal input support (text and images)
  - Unified image generation and editing with Gemini 2.5 Flash Image Preview
  - Intelligent image optimization with size-aware compression algorithms
  - Automated image upload to Open WebUI with robust fallback support
  - Optimized text-to-image and image-to-image workflows
  - Non-streaming mode for image generation to prevent chunk overflow
  - Progressive status updates for optimal user experience
  - Consolidated error handling and comprehensive logging
  - Seamless Google Generative AI and Vertex AI integration
  - Advanced generation parameters (temperature, max tokens, etc.)
  - Configurable safety settings with environment variable support
  - Military-grade encrypted storage of sensitive API keys
  - Intelligent grounding with Google search integration
  - Vertex AI Search grounding for RAG
  - Native tool calling support with automatic signature management
  - URL context grounding for specified web pages
  - Unified image processing with consolidated helper methods
  - Optimized payload creation for image generation models
  - Configurable image processing parameters (size, quality, compression)
  - Flexible upload fallback options and optimization controls
  - Configurable thinking levels (low/high) for Gemini 3 models
  - Configurable thinking budgets (0-32768 tokens) for Gemini 2.5 models
  - Configurable image generation aspect ratio (1:1, 16:9, etc.) and resolution (1K, 2K, 4K)
"""'''


def extract_imports(content: str) -> tuple[set[str], str]:
    """
    Extract import statements from code content.
    Returns a tuple of (import_set, code_without_imports).
    """
    import_pattern = r'^(?:from\s+([\w.]+)\s+import\s+([^\n]+)|import\s+([^\n]+))$'
    
    imports = set()
    lines = content.split('\n')
    non_import_lines = []
    
    for line in lines:
        stripped = line.strip()
        match = re.match(import_pattern, stripped)
        if match:
            imports.add(stripped)
        else:
            non_import_lines.append(line)
    
    return imports, '\n'.join(non_import_lines)


def organize_imports(imports: set[str]) -> str:
    """Organize imports into standard, third-party, and local sections."""
    stdlib = []
    third_party = []
    
    stdlib_modules = {
        'os', 're', 'time', 'asyncio', 'base64', 'hashlib', 'logging', 'io', 'uuid',
        'datetime', 'inspect', 'json', 'traceback', 'typing', 'pathlib'
    }
    
    for imp in sorted(imports):
        if imp.startswith('from .') or imp.startswith('import .'):
            continue
        else:
            module = imp.split()[1].split('.')[0]
            if module in stdlib_modules:
                stdlib.append(imp)
            else:
                third_party.append(imp)
    
    result = []
    if stdlib:
        result.extend(sorted(stdlib))
        result.append('')
    if third_party:
        result.extend(sorted(third_party))
        result.append('')
    
    return '\n'.join(result)


def read_file_content(filepath: Path, remove_docstring: bool = True) -> str:
    """Read file and return content."""
    content = filepath.read_text(encoding='utf-8')
    
    if remove_docstring:
        # Remove module docstring (triple-quoted string at the start)
        content = re.sub(r'^\s*"""[\s\S]*?"""\s*\n', '', content)
    
    return content


def compile_gemini_pipeline():
    """Compile all source files into a single gemini.py file."""
    src_dir = Path('src')
    output_file = Path('gemini_compile.py')
    
    # File order - exclude config.py since its classes are now in pipe.py
    file_order = [
        'constants.py',
        'encryption.py',
        'utils.py',
        'image_processing.py',
        'model_management.py',
        'content_preparation.py',
        'generation_config.py',
        'response_handlers.py',
        'cache.py',
        'image_upload.py',
        'mcp_patch.py',
        'pipe.py',
    ]
    
    all_imports = set()
    all_code_parts = []
    
    # Read and process each file
    for filename in file_order:
        filepath = src_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        print(f"Processing {filename}...")
        content = read_file_content(filepath)
        imports, code = extract_imports(content)
        all_imports.update(imports)
        
        # Add file comment
        all_code_parts.append(f"\n# {'='*20} From {filename} {'='*20}\n")
        all_code_parts.append(code)
    
    # Generate the output
    organized_imports = organize_imports(all_imports)
    
    # Build the output
    output_parts = []
    
    # Add OpenWebUI header first (before any imports)
    output_parts.append(OPENWEBUI_HEADER)
    output_parts.append('\n\n')
    
    # Add imports
    output_parts.append(organized_imports)
    
    # Add all code
    output_parts.append('\n')
    output_parts.append('\n'.join(all_code_parts))
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_parts))
    
    print(f"\nCompilation complete! Output written to: {output_file}")
    
    # Count lines
    with open(output_file, 'r', encoding='utf-8') as f:
        line_count = len(f.readlines())
    print(f"Total lines: {line_count}")


if __name__ == '__main__':
    compile_gemini_pipeline()
