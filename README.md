# ✨ Google Gemini Pipeline for OpenWebUI

A modular, feature-rich pipeline plugin for integrating Google Gemini API with OpenWebUI,  
refactored from the owndev's original implementation to enhance maintainability, performance, and compatibility with MCP tools. 

## 🕊️ Overview

This project is a refactored and enhanced version of the original [Open-WebUI-Functions Google Gemini Pipeline](https://github.com/owndev/Open-WebUI-Functions/blob/main/pipelines/google/google_gemini.py) 🕊️

## 🚀 Key Improvements

### 🌟 Modular Architecture
- **Source code** organized in `src/` directory for maintainability 
- **Compiled output** as single `gemini_compile.py` file for easy deployment 
- Clean separation of concerns with dedicated modules for:
  - Image processing and optimization
  - Model management
  - Content preparation
  - Response handling
  - Encryption utilities

### 📝 System Prompt Caching
- Implements Gemini's context caching API for system instructions
- Reduces token costs for long system prompts
- Automatic cache management with TTL support (1 hour, invalidate on next day for date-sensitive prompts)
- Supports workplace-specific cache keys for different models environments

### 🛠️ MCP Tool Compatibility Fix
- Resolved compatibility issues with MCP (Model Context Protocol) tools
- Proper handling of tool calls and responses in streaming mode

## 🏗️ Architecture

```
src/
├── pipe.py              # Main Pipe class (entry point)
├── cache.py             # System prompt cache manager
├── config.py            # Valves configuration (Pydantic models)
├── constants.py         # Default values and constants
├── content_preparation.py   # Message processing and content prep
├── encryption.py        # API key encryption utilities
├── generation_config.py # Gemini generation configuration
├── image_processing.py  # Image optimization and handling
├── image_upload.py      # Image upload via Files API
├── mcp_patch.py         # MCP tool compatibility patches
├── model_management.py  # Model listing and filtering
├── response_handlers.py # Response streaming and processing
├── token_counter.py     # Token counting utilities
└── utils.py             # Helper functions
```

## 🔄 Build process
compile.py → generates → gemini_compile.py

## 💡 Usage

1. **Development**: Edit files in `src/` directory 
2. **Compile**: Run `python compile.py` to generate `gemini_compile.py` 
3. **Deploy**: Create a new pipeline in OpenWebUI and use the contents of `gemini_compile.py` as the pipeline code

## 📋 Roadmap

### 🚧 In Progress
- **Incremental Conversation Caching**: Smart cache rebuilding based on conversation history to further reduce token costs for long chats 

### ✅ Completed
- [x] Refactoring to modular architecture 
- [x] System prompt caching 
- [x] MCP tool compatibility 
- [x] Comprehensive error handling 
- [x] Bug fixes and optimizations (e.g., strip thoughts from multi-turn conversations) 

## 📜 License

Apache-2.0 license, Same as the original project 🕊️

---

*Made with 💖 and lots of ✨*
