#!/usr/bin/env python3
"""
ISNE Training Status Monitor

Provides overview of training status, recent runs, and system health.

Usage:
  python scripts/check_training_status.py           # Recent status
  python scripts/check_training_status.py --full    # Detailed analysis
  python scripts/check_training_status.py --alerts  # Check for issues
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_recent_training_logs(log_type: str, days: int = 7) -> List[Dict]:
    """Load recent training logs of specified type."""
    
    log_dir = Path(f"./training_logs/{log_type}")
    if not log_dir.exists():
        return []
    
    # Find recent log files
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_logs = []
    
    for log_file in log_dir.glob("*.json"):
        try:
            # Extract timestamp from filename
            if log_type in log_file.stem:
                timestamp_str = log_file.stem.split("_")[-1]
                log_time = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                
                if log_time >= cutoff_date:
                    with open(log_file) as f:
                        log_data = json.load(f)
                        log_data['log_file'] = str(log_file)
                        log_data['log_time'] = log_time.isoformat()
                        recent_logs.append(log_data)
        except Exception:
            continue
    
    return sorted(recent_logs, key=lambda x: x['log_time'], reverse=True)

def analyze_training_trends(logs: List[Dict]) -> Dict[str, Any]:
    """Analyze trends in training performance."""
    
    if not logs:
        return {'status': 'no_data'}
    
    # Extract metrics
    quality_scores = []
    processing_times = []
    chunk_counts = []
    
    for log in logs:
        if 'quality_metrics' in log:
            quality = log['quality_metrics'].get('overall_score')
            if quality is not None:
                quality_scores.append(quality)
        
        if 'processing_time' in log:
            processing_times.append(log['processing_time'])
        
        if 'chunks_processed' in log:
            chunk_counts.append(log['chunks_processed'])
    
    trends = {
        'status': 'analyzed',
        'total_runs': len(logs),
        'date_range': {
            'earliest': logs[-1]['log_time'] if logs else None,
            'latest': logs[0]['log_time'] if logs else None
        }
    }
    
    if quality_scores:
        trends['quality'] = {
            'current': quality_scores[0],
            'average': sum(quality_scores) / len(quality_scores),
            'trend': 'improving' if len(quality_scores) > 1 and quality_scores[0] > quality_scores[-1] else 'stable'
        }
    
    if processing_times:
        trends['performance'] = {
            'avg_time': sum(processing_times) / len(processing_times),
            'recent_time': processing_times[0]
        }
    
    if chunk_counts:
        trends['volume'] = {
            'avg_chunks': sum(chunk_counts) / len(chunk_counts),
            'recent_chunks': chunk_counts[0],
            'total_chunks': sum(chunk_counts)
        }
    
    return trends

def check_system_health() -> Dict[str, Any]:
    """Check overall system health and identify issues."""
    
    health = {
        'status': 'healthy',
        'issues': [],
        'warnings': []
    }
    
    # Check if models exist
    model_paths = [
        Path("./models/current_isne_model.pt"),
        Path("./bootstrap-output/models/refined_isne_model.pt")
    ]
    
    model_found = any(path.exists() for path in model_paths)
    if not model_found:
        health['issues'].append("No trained ISNE model found")
        health['status'] = 'error'
    
    # Check recent training activity
    daily_logs = load_recent_training_logs('daily', days=3)
    if not daily_logs:
        health['warnings'].append("No recent daily training activity")
    
    # Check log directory health
    log_dirs = ['daily', 'weekly', 'monthly']
    for log_type in log_dirs:
        log_dir = Path(f"./training_logs/{log_type}")
        if not log_dir.exists():
            health['warnings'].append(f"Missing {log_type} log directory")
    
    # Check disk space for logs (simplified)
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path("."))
        free_gb = free // (1024**3)
        if free_gb < 5:  # Less than 5GB free
            health['warnings'].append(f"Low disk space: {free_gb}GB free")
    except Exception:
        pass
    
    return health

def print_status_summary(full_detail: bool = False):
    """Print training status summary."""
    
    print("🤖 HADES ISNE Training Status")
    print("=" * 50)
    
    # System health check
    health = check_system_health()
    status_emoji = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'warning' else "❌"
    print(f"{status_emoji} System Health: {health['status'].upper()}")
    
    if health['issues']:
        print("  Issues:")
        for issue in health['issues']:
            print(f"    ❌ {issue}")
    
    if health['warnings']:
        print("  Warnings:")
        for warning in health['warnings']:
            print(f"    ⚠️ {warning}")
    
    print()
    
    # Recent training activity
    training_types = ['daily', 'weekly', 'monthly']
    
    for training_type in training_types:
        print(f"📊 {training_type.title()} Training")
        print("-" * 30)
        
        logs = load_recent_training_logs(training_type, days=7 if training_type == 'daily' else 30)
        
        if not logs:
            print(f"  No recent {training_type} training logs found")
            print()
            continue
        
        trends = analyze_training_trends(logs)
        
        print(f"  Recent runs: {trends['total_runs']}")
        if trends['date_range']['latest']:
            latest_time = datetime.fromisoformat(trends['date_range']['latest'])
            time_ago = datetime.now() - latest_time
            print(f"  Last run: {time_ago.days} days ago")
        
        if 'quality' in trends:
            quality = trends['quality']
            print(f"  Quality score: {quality['current']:.3f} (avg: {quality['average']:.3f})")
        
        if 'volume' in trends and full_detail:
            volume = trends['volume']
            print(f"  Recent chunks: {volume['recent_chunks']}")
            print(f"  Total processed: {volume['total_chunks']}")
        
        if 'performance' in trends and full_detail:
            perf = trends['performance']
            print(f"  Processing time: {perf['recent_time']:.1f}s (avg: {perf['avg_time']:.1f}s)")
        
        print()
    
    # Model status
    print("🔧 Model Status")
    print("-" * 20)
    
    model_paths = [
        ("Current Model", Path("./models/current_isne_model.pt")),
        ("Bootstrap Model", Path("./bootstrap-output/models/refined_isne_model.pt")),
        ("Weekly Backup", Path("./training_logs/weekly/latest_model.pt"))
    ]
    
    for name, path in model_paths:
        if path.exists():
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            age = datetime.now() - mod_time
            print(f"  ✅ {name}: {age.days} days old")
        else:
            print(f"  ❌ {name}: Not found")
    
    print()

def main():
    """Main status check function."""
    
    parser = argparse.ArgumentParser(description="Check ISNE training status")
    parser.add_argument("--full", action="store_true", 
                       help="Show detailed analysis")
    parser.add_argument("--alerts", action="store_true",
                       help="Check for alerts and issues only")
    parser.add_argument("--json", action="store_true",
                       help="Output status as JSON")
    
    args = parser.parse_args()
    
    if args.json:
        # JSON output for scripting
        status_data = {
            'health': check_system_health(),
            'training_summary': {}
        }
        
        for training_type in ['daily', 'weekly', 'monthly']:
            logs = load_recent_training_logs(training_type, days=30)
            status_data['training_summary'][training_type] = analyze_training_trends(logs)
        
        print(json.dumps(status_data, indent=2, default=str))
        return 0
    
    if args.alerts:
        # Alert mode - only show issues
        health = check_system_health()
        if health['status'] != 'healthy':
            print(f"ALERT: System status is {health['status']}")
            for issue in health['issues']:
                print(f"ERROR: {issue}")
            for warning in health['warnings']:
                print(f"WARNING: {warning}")
            return 1
        else:
            print("All systems operational")
            return 0
    
    # Normal status display
    print_status_summary(full_detail=args.full)
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)