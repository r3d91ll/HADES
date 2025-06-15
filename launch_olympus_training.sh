#!/bin/bash

# Olympus Production RAG Training Script
# Trains comprehensive ISNE model with PDFs + curated Python codebase

echo "🚀 Starting Olympus Production RAG Training..."
echo "================================================"

# Activate virtual environment
source venv/bin/activate

# Collect input files
echo "📁 Collecting training files..."
PDF_COUNT=$(find ../test-data3 -name "*.pdf" | wc -l)
PY_COUNT=$(find src -type f -name "*.py" -size +500c | wc -l)
LADON_COUNT=$(find ../ladon -type f -name "*.py" -size +100c | wc -l)

echo "   📄 PDFs found: $PDF_COUNT"
echo "   🐍 Python files (HADES): $PY_COUNT" 
echo "   🐍 Python files (ladon): $LADON_COUNT"

# Create temporary file lists to avoid command line length issues
echo "   📝 Creating file lists..."
find ../test-data3 -name "*.pdf" | head -200 > /tmp/pdf_list.txt
find src -type f -name "*.py" -size +500c | head -100 > /tmp/py_list.txt
find ../ladon -type f -name "*.py" -size +100c > /tmp/ladon_list.txt

# Count selected files
PDF_SELECTED=$(wc -l < /tmp/pdf_list.txt)
PY_SELECTED=$(wc -l < /tmp/py_list.txt) 
LADON_SELECTED=$(wc -l < /tmp/ladon_list.txt)

echo "   ✅ Selected: $PDF_SELECTED PDFs + $PY_SELECTED HADES Python files + $LADON_SELECTED ladon Python files"

# Launch training
echo ""
echo "🎯 Launching GPU-accelerated training pipeline..."
echo "================================================"

# Create a combined file list and use the --input-files-list option to avoid CLI parsing issues
echo "   🎯 Creating combined file list..."
cat /tmp/pdf_list.txt /tmp/py_list.txt /tmp/ladon_list.txt | head -100 > /tmp/combined_files.txt
TOTAL_FILES=$(wc -l < /tmp/combined_files.txt)
echo "   🎯 Total files selected: $TOTAL_FILES"

# Use the new --input-files-list option to pass file list
python -m src.isne.bootstrap.cli \
  --input-files-list /tmp/combined_files.txt \
  --output-dir ./output/olympus_production \
  --model-name "olympus_production_rag_model" \
  --monitor \
  --log-level INFO \
  --override chunking.chunker_type=core \
  --override embedding.device=cuda \
  --override embedding.embedder_type=gpu \
  --override isne_training.device=cuda \
  --override embedding.batch_size=128 \
  --override isne_training.batch_size=128 \
  --override graph_construction.similarity_threshold=0.75 \
  --override graph_construction.max_edges_per_node=8 \
  --override isne_training.epochs=25

# Cleanup
rm -f /tmp/pdf_list.txt /tmp/py_list.txt /tmp/ladon_list.txt /tmp/combined_files.txt

echo ""
echo "✅ Training completed! Check output/olympus_production/ for results."
