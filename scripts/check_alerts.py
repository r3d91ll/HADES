#!/usr/bin/env python
"""
CLI utility for checking and managing HADES-PathRAG alerts.

This script provides a command-line interface for reviewing
alerts, filtering by severity, and performing basic alert
management operations.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts import AlertManager, AlertLevel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HADES-PathRAG Alert Checker"
    )
    
    parser.add_argument(
        "--alert-dir",
        type=str,
        default="./alerts",
        help="Directory containing alert logs"
    )
    
    parser.add_argument(
        "--min-level",
        type=str,
        choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default="LOW",
        help="Minimum alert level to display"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        help="Filter alerts by source"
    )
    
    parser.add_argument(
        "--last",
        type=str,
        help="Show alerts from last period (e.g., '1h', '2d', '30m')"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of alerts to display"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show alert statistics"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear alerts after viewing"
    )
    
    return parser.parse_args()


def load_alerts(alert_dir: str) -> List[Dict[str, Any]]:
    """Load alerts from the JSON log file."""
    json_file = Path(alert_dir) / "alerts.json"
    
    if not json_file.exists():
        return []
    
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not parse alerts file {json_file}")
        return []


def filter_alerts(
    alerts: List[Dict[str, Any]],
    min_level: str,
    source: Optional[str] = None,
    time_period: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Filter alerts based on criteria."""
    # Map level names to numeric values for comparison
    level_values = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4
    }
    
    min_level_value = level_values.get(min_level, 1)
    
    # Filter by level
    filtered = [a for a in alerts if level_values.get(a["level"], 0) >= min_level_value]
    
    # Filter by source if specified
    if source:
        filtered = [a for a in filtered if source.lower() in a["source"].lower()]
    
    # Filter by time period if specified
    if time_period:
        # Parse time period string (e.g., "1h", "2d", "30m")
        try:
            unit = time_period[-1].lower()
            value = int(time_period[:-1])
            
            now = datetime.now()
            
            if unit == "h":
                cutoff_time = now - timedelta(hours=value)
            elif unit == "d":
                cutoff_time = now - timedelta(days=value)
            elif unit == "m":
                cutoff_time = now - timedelta(minutes=value)
            else:
                raise ValueError(f"Unknown time unit: {unit}")
            
            cutoff_timestamp = cutoff_time.timestamp()
            filtered = [a for a in filtered if a["timestamp"] >= cutoff_timestamp]
            
        except (ValueError, IndexError):
            print(f"Warning: Could not parse time period '{time_period}', ignoring this filter")
    
    return filtered


def print_alerts_text(alerts: List[Dict[str, Any]], limit: int):
    """Print alerts in text format."""
    if not alerts:
        print("No alerts found matching the criteria.")
        return
    
    # Display alerts (newest first)
    for i, alert in enumerate(alerts[:limit]):
        level = alert["level"]
        
        # Format based on level
        if level == "CRITICAL":
            level_str = f"\033[1;31m{level}\033[0m"  # Bold red
        elif level == "HIGH":
            level_str = f"\033[31m{level}\033[0m"    # Red
        elif level == "MEDIUM":
            level_str = f"\033[33m{level}\033[0m"    # Yellow
        else:
            level_str = f"\033[32m{level}\033[0m"    # Green
        
        # Print alert details
        print(f"{i+1}. [{alert['timestamp_formatted']}] {level_str}: {alert['message']}")
        print(f"   Source: {alert['source']}")
        
        # Print context if available
        if alert.get("context"):
            print("   Context:")
            for key, value in alert["context"].items():
                if isinstance(value, dict):
                    print(f"     {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"       {sub_key}: {sub_value}")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"     {key}: [{len(value)} items]")
                else:
                    print(f"     {key}: {value}")
        
        print()
    
    # Show count if limited
    if len(alerts) > limit:
        print(f"... and {len(alerts) - limit} more alerts")


def print_alerts_json(alerts: List[Dict[str, Any]], limit: int):
    """Print alerts in JSON format."""
    print(json.dumps(alerts[:limit], indent=2))


def print_alert_stats(alerts: List[Dict[str, Any]]):
    """Print alert statistics."""
    if not alerts:
        print("No alerts found.")
        return
    
    # Count by level
    level_counts = {}
    for alert in alerts:
        level = alert["level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    
    # Count by source
    source_counts = {}
    for alert in alerts:
        source = alert["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Find time range
    timestamps = [a["timestamp"] for a in alerts]
    oldest = datetime.fromtimestamp(min(timestamps))
    newest = datetime.fromtimestamp(max(timestamps))
    time_span = newest - oldest
    
    # Print stats
    print("\n===== Alert Statistics =====")
    print(f"Total alerts: {len(alerts)}")
    print(f"Time span: {time_span}")
    print(f"Oldest: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Newest: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nAlert levels:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = level_counts.get(level, 0)
        print(f"  {level}: {count}")
    
    print("\nTop sources:")
    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    for source, count in sorted_sources[:5]:
        print(f"  {source}: {count}")
    
    print("===========================\n")


def clear_alerts(alert_dir: str):
    """Clear all alerts."""
    json_file = Path(alert_dir) / "alerts.json"
    log_file = Path(alert_dir) / "alerts.log"
    critical_log_file = Path(alert_dir) / "critical_alerts.log"
    
    # Clear JSON file
    if json_file.exists():
        with open(json_file, "w") as f:
            f.write("[]")
    
    # Clear log files
    for file in [log_file, critical_log_file]:
        if file.exists():
            with open(file, "w") as f:
                f.write("")
    
    print("All alerts have been cleared.")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create alert directory if it doesn't exist
    Path(args.alert_dir).mkdir(parents=True, exist_ok=True)
    
    # Load alerts
    alerts = load_alerts(args.alert_dir)
    
    # Filter alerts
    filtered_alerts = filter_alerts(
        alerts,
        args.min_level,
        args.source,
        args.last
    )
    
    # Show statistics if requested
    if args.stats:
        print_alert_stats(filtered_alerts)
    
    # Display filtered alerts
    if args.format == "text":
        print_alerts_text(filtered_alerts, args.limit)
    else:
        print_alerts_json(filtered_alerts, args.limit)
    
    # Clear alerts if requested
    if args.clear:
        clear_alerts(args.alert_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
