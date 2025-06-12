#!/bin/bash
"""
ISNE Training Cron Job Setup Script

Sets up hierarchical ISNE training schedule:
- Daily incremental training at 2 AM
- Weekly full retraining on Sundays at 3 AM  
- Monthly comprehensive retraining on 1st of month at 4 AM

Usage:
  ./scripts/setup_training_cron.sh
"""

# Get absolute path to project
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Setting up ISNE training cron jobs for project: $PROJECT_ROOT"

# Create cron job entries
CRON_ENTRIES=$(cat <<EOF
# HADES ISNE Training Schedule
# Generated on $(date)

# Daily incremental training at 2 AM (process yesterday's data)
0 2 * * * cd $PROJECT_ROOT && python scripts/daily_incremental_training.py >> ./training_logs/daily/cron.log 2>&1

# Weekly full retraining on Sundays at 3 AM
0 3 * * 0 cd $PROJECT_ROOT && python scripts/weekly_full_retraining.py --scope weekly >> ./training_logs/weekly/cron.log 2>&1

# Monthly comprehensive retraining on 1st of month at 4 AM
0 4 1 * * cd $PROJECT_ROOT && python scripts/weekly_full_retraining.py --scope monthly >> ./training_logs/monthly/cron.log 2>&1

EOF
)

# Create log directories
mkdir -p "$PROJECT_ROOT/training_logs/daily"
mkdir -p "$PROJECT_ROOT/training_logs/weekly" 
mkdir -p "$PROJECT_ROOT/training_logs/monthly"

echo "Cron job entries to add:"
echo "$CRON_ENTRIES"
echo ""

# Check if cron is available
if ! command -v crontab &> /dev/null; then
    echo "ERROR: crontab command not found. Please install cron."
    exit 1
fi

# Backup existing crontab
echo "Backing up existing crontab..."
crontab -l > "$PROJECT_ROOT/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || echo "No existing crontab found"

# Ask for confirmation
read -p "Add these cron jobs to your crontab? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRIES") | crontab -
    echo "✅ Cron jobs added successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep -A5 -B1 "HADES"
else
    echo "Cron jobs not added. You can manually add them later:"
    echo "$CRON_ENTRIES"
fi

echo ""
echo "Training schedule setup complete!"
echo ""
echo "Manual commands:"
echo "  Daily:   python scripts/daily_incremental_training.py"
echo "  Weekly:  python scripts/weekly_full_retraining.py --scope weekly"
echo "  Monthly: python scripts/weekly_full_retraining.py --scope monthly"
echo ""
echo "Log locations:"
echo "  Daily:   $PROJECT_ROOT/training_logs/daily/"
echo "  Weekly:  $PROJECT_ROOT/training_logs/weekly/"
echo "  Monthly: $PROJECT_ROOT/training_logs/monthly/"