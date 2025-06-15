#!/bin/bash

# ISNE Model Evaluation Runner
# Automatically evaluates the most recent ISNE training output

echo "🔍 ISNE Model Evaluation Runner"
echo "================================"

# Check if training output exists
OUTPUT_DIR="./output/olympus_production"
MODEL_PATH="$OUTPUT_DIR/isne_model_final.pth"
GRAPH_DATA="$OUTPUT_DIR/graph_data.json"
EVAL_DIR="./evaluation_results/$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo "   Please ensure ISNE training has completed successfully."
    exit 1
fi

if [ ! -f "$GRAPH_DATA" ]; then
    echo "❌ Graph data file not found: $GRAPH_DATA"
    echo "   Please ensure graph construction completed successfully."
    exit 1
fi

echo "✅ Found model: $MODEL_PATH"
echo "✅ Found graph data: $GRAPH_DATA"
echo "📊 Evaluation results will be saved to: $EVAL_DIR"
echo ""

# Create evaluation directory
mkdir -p "$EVAL_DIR"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install additional dependencies if needed
echo "📦 Installing evaluation dependencies..."
pip install scikit-learn matplotlib seaborn > /dev/null 2>&1

# Run evaluation
echo "🚀 Starting ISNE model evaluation..."
echo "   This may take several minutes depending on model size..."
echo ""

python scripts/evaluate_isne_model.py \
    --model-path "$MODEL_PATH" \
    --graph-data "$GRAPH_DATA" \
    --output-dir "$EVAL_DIR"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "📊 Results available in: $EVAL_DIR"
    echo ""
    echo "📁 Generated files:"
    ls -la "$EVAL_DIR"
    echo ""
    echo "📋 Quick summary:"
    if [ -f "$EVAL_DIR/evaluation_results.json" ]; then
        echo "   - evaluation_results.json: Complete evaluation metrics"
    fi
    if [ -f "$EVAL_DIR/tsne_comparison.png" ]; then
        echo "   - tsne_comparison.png: t-SNE visualization of embeddings"
    fi
    if [ -f "$EVAL_DIR/embedding_distributions.png" ]; then
        echo "   - embedding_distributions.png: Embedding distribution analysis"
    fi
    if [ -f "$EVAL_DIR/degree_distribution.png" ]; then
        echo "   - degree_distribution.png: Graph structure analysis"
    fi
else
    echo ""
    echo "❌ Evaluation failed with exit code: $EVAL_EXIT_CODE"
    echo "   Check the logs above for error details."
    exit $EVAL_EXIT_CODE
fi