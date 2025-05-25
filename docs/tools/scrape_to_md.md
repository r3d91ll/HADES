# Scrape to Markdown (scrape_to_md.py)

## Overview

`scrape_to_md.py` is a powerful web scraping utility that converts web pages into clean, well-formatted markdown documents. It uses LLM-assisted extraction to intelligently process the content, preserving meaningful structure while discarding navigation elements, ads, and other non-essential content.

The tool supports both single-page scraping and recursive crawling with configurable depth, making it suitable for creating comprehensive markdown documentation from websites.

## Requirements

### Dependencies

```bash
requests>=2.28.0
beautifulsoup4>=4.11.0
openai>=1.0.0
lxml>=4.9.0
tqdm>=4.64.0
```

### Optional Dependencies

- **vLLM Support**: For high-performance, cost-effective LLM processing

  ```bash
  vllm>=0.2.0
  ```

- **Playwright** (for JavaScript-heavy sites)

  ```bash
  playwright>=1.32.0
  ```

## Integration with vLLM

`scrape_to_md.py` seamlessly integrates with vLLM for high-throughput, efficient processing of web content. This integration:

1. **Reduces costs**: By using local models instead of API calls
2. **Increases speed**: vLLM's optimized batch processing enables faster content extraction
3. **Enhances privacy**: Keeps sensitive data within your infrastructure

To use vLLM:

1. Ensure vLLM is installed: `pip install vllm`
2. Start a vLLM server with your preferred model:

   ```bash
   python -m vllm.entrypoints.openai.api_server --model <your-model> --port 8000
   ```

3. Configure the tool to use your local vLLM endpoint:

   ```bash
   python tools/scrape_to_md.py --base-url http://localhost:8000/v1 --api-key dummy <url>
   ```

4. example vLLM command:

   ```bash
   CUDA_VISIBLE_DEVICES=1 vllm serve Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1 --enable-reasoning --reasoning-parser deepseek_r1 --max-model-len 40960 --host 0.0.0.0 --port 8345 --served-model-name Josiefied-Qwen3-8B-abliterated-v1
   ```

   note: the --served-model-name flag is important for the script to properly identify the model

## How It Works

1. **Fetching**: Retrieves web pages using requests or Playwright (for JavaScript-heavy sites)
2. **Processing**:
   - Extracts HTML content
   - Uses BeautifulSoup to perform initial cleanup
   - Applies LLM-guided extraction to identify and format key content
3. **Conversion**: Structures the output as clean markdown
4. **Storage**: Saves processed markdown files in the specified output directory

The LLM is used to intelligently extract meaningful content while discarding navigation, ads, and other non-essential elements, resulting in clean, content-focused markdown.

## Usage Examples

### Basic Scraping

```bash
# Scrape a single page
python tools/scrape_to_md.py https://example.com/page

# Specify output directory
python tools/scrape_to_md.py -o ./docs/extracted https://example.com/page

# Custom extraction prompt
python tools/scrape_to_md.py -p "Extract the main content and preserve code examples" https://example.com/page
```

### Batch Processing

```bash
# Scrape URLs from a file
python tools/scrape_to_md.py -f urls.txt

# Process multiple URLs
python tools/scrape_to_md.py https://example.com/page1 https://example.com/page2
```

### Web Crawling

```bash
# Crawl a site with default depth (2)
python tools/scrape_to_md.py --crawl https://example.com

# Specify crawl depth
python tools/scrape_to_md.py --crawl --depth 3 https://example.com

# Limit the number of pages
python tools/scrape_to_md.py --crawl --max-pages 50 https://example.com
```

### Model Configuration

```bash
# Use a specific OpenAI model
python tools/scrape_to_md.py -m gpt-4 https://example.com

# Use local vLLM server
python tools/scrape_to_md.py --base-url http://localhost:8000/v1 --api-key dummy -m mistral-7b https://example.com

# Use Ollama
python tools/scrape_to_md.py --base-url http://localhost:11434/v1 --api-key ollama --raw-model mistral https://example.com
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `urls` | One or more URLs to scrape |
| `-f, --file` | File containing URLs to scrape (one per line) |
| `-o, --outdir` | Output directory for markdown files (default: ./markdown_output) |
| `-p, --prompt` | Custom extraction prompt (default: "Extract the title and main body text.") |
| `-m, --model` | Model to use for extraction (default: gpt-4o-mini) |
| `--base-url` | Base URL for API requests (for vLLM/self-hosted endpoints) |
| `-k, --api-key` | OpenAI API key (defaults to OPENAI_API_KEY env var) |
| `--raw-model` | Send model string exactly without "openai/" prefix |
| `--model-tokens` | Context length of the model |
| `--no-headless` | Disable headless mode for browser-based scraping |
| `-v, --verbose` | Enable verbose output |
| `--crawl` | Enable crawling mode |
| `--depth` | Maximum crawl depth (default: 2) |
| `--max-pages` | Maximum number of pages to crawl (default: 100) |
| `--same-domain-only` | Only follow links to the same domain (default: True) |
| `--ignore-robots` | Ignore robots.txt restrictions |

## Examples for ArangoDB Documentation

To scrape ArangoDB documentation for PathRAG ingestion:

```bash
# Using OpenAI
python tools/scrape_to_md.py --crawl --depth 3 --max-pages 200 \
  -o ./test-output/markdown_output/docs-arangodb-com \
  -p "Extract the complete documentation content, preserving headings, code blocks, and technical details." \
  -m gpt-4o-mini \
  https://docs.arangodb.com/3.11/

# Using local vLLM with Mixtral
python tools/scrape_to_md.py --crawl --depth 3 --max-pages 200 \
  -o ./test-output/markdown_output/docs-arangodb-com \
  -p "Extract the complete documentation content, preserving headings, code blocks, and technical details." \
  -m mixtral-8x7b \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  https://docs.arangodb.com/3.11/
```

## Troubleshooting

- **Rate Limiting**: When using OpenAI, you may encounter rate limits. Use `--max-pages` to control batch size.
- **Memory Issues**: For large sites, crawling may consume significant memory. Use smaller depth values or limit pages.
- **JavaScript-Heavy Sites**: Some sites require JavaScript rendering. Ensure Playwright is installed and avoid using `--no-headless`.
- **Model Context Limits**: For very large pages, set an appropriate model with `--model-tokens` parameter.

## Testing and Quality Assurance

The `scrape_to_md.py` tool includes a comprehensive test suite to ensure reliable operation:

- **Unit Tests**: Cover core functionality including URL normalization, link extraction, and crawling logic
- **Integration Tests**: Verify end-to-end operation with mock web responses
- **Type Checking**: All functions are fully typed and verified with mypy

To run tests:

```bash
# Run unit tests
python -m pytest tests/tools/test_scrape_to_md.py

# Check type consistency
mypy tools/scrape_to_md.py
```

### Current Test Coverage

The tool maintains >85% test coverage, with particular focus on the core extraction and processing functions.
