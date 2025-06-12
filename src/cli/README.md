# CLI Module

The CLI module provides specialized command-line interfaces for administrative operations, document ingestion, and system querying. These CLI tools are designed for system administrators, researchers, and power users who need direct access to HADES functionality without going through the REST API.

## 📋 Overview

This module implements task-specific command-line interfaces that:

- Provide direct access to core system functionality
- Enable batch operations and automation scripting
- Offer administrative tools for system management
- Support both interactive and non-interactive usage modes
- Integrate with shell scripting and automation workflows

## 🏗️ Architecture

### CLI Organization

```text
CLI Commands → Task-Specific Handlers → Core System Components
     ↓
Administrative Tools (admin.py)
Ingestion Tools (ingest.py)  
Query Tools (query.py)
```

### Design Philosophy

- **Task-focused**: Each CLI module handles a specific operational domain
- **Composable**: Commands can be chained and scripted
- **Configuration-aware**: Inherits from system configuration files
- **Error-resilient**: Graceful error handling and recovery
- **Progress-aware**: Real-time feedback for long-running operations

## 📁 Module Contents

### Core CLI Files

- **`admin.py`** - Administrative commands for system management
- **`ingest.py`** - Document ingestion and processing workflows
- **`query.py`** - Interactive and batch querying capabilities
- **`__init__.py`** - Module exports and shared CLI utilities

## 🛠️ Administrative CLI (`admin.py`)

### System Status and Health

**Check overall system health**:

```bash
python -m src.cli.admin status
```

**Detailed system diagnostics**:

```bash
python -m src.cli.admin health-check --verbose
```

**Database connectivity and status**:

```bash
python -m src.cli.admin db-status
```

### Training Management

**View training history**:

```bash
python -m src.cli.admin training-history --days 30
```

**Force training operations**:

```bash
python -m src.cli.admin force-training --type weekly --scope weekly
```

**Model management**:

```bash
# List available models
python -m src.cli.admin list-models

# Switch active model
python -m src.cli.admin set-model --path ./models/refined_isne_model.pt

# Backup current model
python -m src.cli.admin backup-model --output ./backups/
```

### System Configuration

**View current configuration**:

```bash
python -m src.cli.admin config-show
```

**Validate configuration files**:

```bash
python -m src.cli.admin config-validate --config-dir ./src/config/
```

**Update configuration**:

```bash
python -m src.cli.admin config-update --key embedding.model --value modernbert
```

### Maintenance Operations

**Clear system caches**:

```bash
python -m src.cli.admin clear-cache --type all
```

**Database maintenance**:

```bash
# Vacuum and optimize database
python -m src.cli.admin db-maintenance

# Rebuild indexes
python -m src.cli.admin rebuild-indexes
```

**Log management**:

```bash
# Archive old logs
python -m src.cli.admin archive-logs --older-than 30

# Clean up temporary files
python -m src.cli.admin cleanup-temp
```

## 📥 Ingestion CLI (`ingest.py`)

### Document Processing

**Single document ingestion**:

```bash
python -m src.cli.ingest document --input ./paper.pdf --output ./processed/
```

**Batch directory processing**:

```bash
python -m src.cli.ingest batch \
  --input-dir ./corpus/ \
  --output-dir ./processed/ \
  --pattern "*.pdf" \
  --recursive
```

**Code repository ingestion**:

```bash
python -m src.cli.ingest repo \
  --repo-path ./src/ \
  --output ./processed_code/ \
  --include-tests \
  --preserve-structure
```

### Processing Configuration

**Custom chunking strategies**:

```bash
python -m src.cli.ingest document \
  --input ./document.pdf \
  --chunking-strategy adaptive \
  --chunk-size 512 \
  --overlap 64
```

**Embedding configuration**:

```bash
python -m src.cli.ingest batch \
  --input-dir ./docs/ \
  --embedding-model modernbert \
  --batch-size 32 \
  --gpu-acceleration
```

### Progress Monitoring

**Real-time progress display**:

```bash
python -m src.cli.ingest batch \
  --input-dir ./large_corpus/ \
  --progress-bar \
  --verbose \
  --log-file ./ingestion.log
```

**Resume interrupted processing**:

```bash
python -m src.cli.ingest resume \
  --job-id ingestion_20240115_143022 \
  --output-dir ./processed/
```

### Quality Validation

**Validate processed documents**:

```bash
python -m src.cli.ingest validate \
  --processed-dir ./processed/ \
  --check-embeddings \
  --check-relationships
```

**Generate processing reports**:

```bash
python -m src.cli.ingest report \
  --processed-dir ./processed/ \
  --output ./reports/ingestion_report.json \
  --include-stats
```

## 🔍 Query CLI (`query.py`)

### Interactive Querying

**Start interactive query session**:

```bash
python -m src.cli.query interactive
```

```text
HADES Query Interface
=============================
Enter your questions (type 'exit' to quit, 'help' for commands)

> How does the chunking system work?
🔍 Searching knowledge base...

Answer: The chunking system in HADES uses adaptive strategies...

Sources:
- chunking/base.py (score: 0.92)
- chunking/ast_chunker.py (score: 0.89)

> help
Available commands:
  set mode [naive|local|global|hybrid] - Change query mode
  set results <n> - Set maximum results
  sources on|off - Toggle source display
  export <filename> - Export session to file
  
> exit
Session exported to query_session_20240115.json
```

### Batch Querying

**Process questions from file**:

```bash
python -m src.cli.query batch \
  --questions ./questions.txt \
  --output ./answers.json \
  --mode hybrid \
  --max-results 5
```

**Question file format** (`questions.txt`):

```text
How does ISNE training work?
What are the key components of document processing?
Explain the PathRAG retrieval mechanism
How is code chunking different from text chunking?
```

### Query Modes and Options

**Different retrieval strategies**:

```bash
# Naive similarity search
python -m src.cli.query ask "Question" --mode naive

# Local graph traversal  
python -m src.cli.query ask "Question" --mode local --depth 2

# Global graph analysis
python -m src.cli.query ask "Question" --mode global

# Hybrid approach (recommended)
python -m src.cli.query ask "Question" --mode hybrid
```

**Filtering and constraints**:

```bash
python -m src.cli.query ask "Question" \
  --filter-type python \
  --filter-date-after 2024-01-01 \
  --min-score 0.7 \
  --max-results 10
```

### Query Analysis

**Explain query processing**:

```bash
python -m src.cli.query explain "How does chunking work?" \
  --show-steps \
  --include-timing \
  --verbose
```

**Query performance analysis**:

```bash
python -m src.cli.query benchmark \
  --questions ./test_questions.txt \
  --iterations 5 \
  --output ./benchmark_results.json
```

## 🔧 Configuration and Customization

### CLI Configuration

**Global CLI configuration** (`~/.hades_cli.yaml`):

```yaml
cli:
  default_output_format: json  # json, table, plain
  progress_bars: true
  color_output: true
  log_level: INFO

ingest:
  default_chunking_strategy: adaptive
  default_batch_size: 16
  auto_validate: true

query:
  default_mode: hybrid
  default_max_results: 10
  show_sources: true
  
admin:
  confirm_destructive: true
  backup_before_changes: true
```

### Environment Variables

**Configure CLI behavior**:

```bash
# Output formatting
export HADES_CLI_FORMAT=json
export HADES_CLI_COLOR=true

# Default paths
export HADES_MODEL_PATH=./models/current_isne_model.pt
export HADES_CONFIG_PATH=./src/config/

# Performance tuning
export HADES_CLI_PARALLEL_JOBS=4
export HADES_CLI_TIMEOUT=300
```

### Custom Commands

**Extend CLI with custom commands**:

```python
# custom_commands.py
import click
from src.cli.admin import admin_cli

@admin_cli.command()
@click.option('--project-path', required=True)
@click.option('--output-dir', required=True) 
def analyze_project(project_path, output_dir):
    """Analyze a code project and generate comprehensive report."""
    from src.analysis import ProjectAnalyzer
    
    analyzer = ProjectAnalyzer()
    report = analyzer.analyze(project_path)
    report.save(output_dir)
    click.echo(f"Analysis complete: {output_dir}/project_report.json")
```

## 📊 Output Formats and Integration

### Output Formats

**JSON output for scripting**:

```bash
python -m src.cli.query ask "Question" --format json | jq '.answer'
```

**Table format for human reading**:

```bash
python -m src.cli.admin training-history --format table
```

**Plain text for logging**:

```bash
python -m src.cli.ingest batch --input-dir ./docs/ --format plain >> ingestion.log
```

### Shell Integration

**Shell completion setup**:

```bash
# Bash completion
eval "$(_HADES_CLI_COMPLETE=bash_source python -m src.cli.admin)"

# Zsh completion  
eval "$(_HADES_CLI_COMPLETE=zsh_source python -m src.cli.admin)"
```

**Pipeline integration**:

```bash
# Process and immediately query
python -m src.cli.ingest document --input paper.pdf --output ./temp/ && \
python -m src.cli.query ask "Summarize this paper" --format plain
```

### Automation Scripts

**Example automation script**:

```bash
#!/bin/bash
# daily_processing.sh

set -e

# Process new documents
echo "Processing new documents..."
python -m src.cli.ingest batch \
  --input-dir ./incoming/ \
  --output-dir ./processed/ \
  --move-processed

# Run quality validation
echo "Validating processed documents..."
python -m src.cli.ingest validate \
  --processed-dir ./processed/ \
  --min-quality 0.8

# Generate daily report
echo "Generating daily report..."
python -m src.cli.admin training-history \
  --days 1 \
  --format json > ./reports/daily_$(date +%Y%m%d).json

echo "Daily processing complete!"
```

## 🔍 Debugging and Troubleshooting

### Verbose Mode and Logging

**Enable detailed logging**:

```bash
python -m src.cli.ingest batch \
  --input-dir ./docs/ \
  --verbose \
  --log-level DEBUG \
  --log-file ./debug.log
```

**Trace command execution**:

```bash
python -m src.cli.query ask "Question" \
  --trace \
  --explain \
  --timing
```

### Error Recovery

**Resume failed operations**:

```bash
# Check for interrupted jobs
python -m src.cli.admin list-jobs --status failed

# Resume specific job
python -m src.cli.ingest resume --job-id <job_id>
```

**Validate system state**:

```bash
# Comprehensive system validation
python -m src.cli.admin validate-system \
  --check-models \
  --check-database \
  --check-config \
  --fix-issues
```

### Performance Tuning

**Optimize for large datasets**:

```bash
python -m src.cli.ingest batch \
  --input-dir ./large_corpus/ \
  --parallel-jobs 8 \
  --memory-limit 16GB \
  --checkpoint-interval 100
```

## 📚 Related Documentation

- **API Module**: See `api/README.md` for programmatic interfaces
- **Configuration**: See `config/README.md` for configuration options
- **Pipeline Operations**: See `orchestration/README.md` for pipeline details
- **Training Management**: See `isne/bootstrap/isne_bootstrap.md` for training workflows

## 🎯 Best Practices

1. **Use appropriate output formats** - JSON for scripting, table for humans
2. **Leverage configuration files** - Avoid repeating command-line options
3. **Monitor long operations** - Use progress bars and logging for batch jobs
4. **Validate inputs** - Check file existence and format before processing
5. **Handle errors gracefully** - Use error codes and meaningful messages
6. **Document custom commands** - Add help text and examples
7. **Test automation scripts** - Verify scripts work with various inputs
8. **Use shell completion** - Improve productivity with auto-completion
