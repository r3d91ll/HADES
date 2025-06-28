#!/usr/bin/env python3
"""
RAID1 Thermal Monitor for ML Processing

Specifically monitors the two drives in md1:
- nvme0n1 (nvme-pci-8100) - Crucial T700 4TB
- nvme1n1 (nvme-pci-8200) - Crucial T700 4TB
"""

import subprocess
import json
import time
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional


class RAID1Monitor:
    def __init__(self) -> None:
        # Map sensors to drives based on PCI addresses for RAID1
        self.drive_sensors = {
            'nvme0n1': 'nvme-pci-8100',  # Crucial T700 4TB
            'nvme1n1': 'nvme-pci-8200'   # Crucial T700 4TB
        }
        
        self.thresholds = {
            'normal': 60,
            'warm': 70,
            'hot': 80,
            'critical': 85
        }
    
    def get_temperatures(self) -> Dict[str, float]:
        """Get current temperatures for RAID1 drives."""
        temps = {}
        
        try:
            result = subprocess.run(['sensors', '-j'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                sensor_data = json.loads(result.stdout)
                
                for drive, sensor in self.drive_sensors.items():
                    if sensor in sensor_data:
                        sensor_info = sensor_data[sensor]
                        # Find Composite temperature
                        for key, value in sensor_info.items():
                            if 'Composite' in key and isinstance(value, dict):
                                temp_key = next((k for k in value.keys() if 'input' in k), None)
                                if temp_key:
                                    temps[drive] = float(value[temp_key])
                                    break
        except Exception as e:
            print(f"Error reading sensors: {e}")
        
        return temps
    
    def get_status(self, temp: float) -> Tuple[str, str]:
        """Get status for a temperature."""
        if temp >= self.thresholds['critical']:
            return 'CRITICAL', '🚨'
        elif temp >= self.thresholds['hot']:
            return 'HOT', '🔥'
        elif temp >= self.thresholds['warm']:
            return 'WARM', '⚠️'
        else:
            return 'NORMAL', '✅'
    
    def print_status(self) -> float:
        """Print current RAID1 thermal status."""
        temps = self.get_temperatures()
        
        if not temps:
            print("❌ Unable to read temperatures")
            return 0.0
        
        max_temp = max(temps.values())
        avg_temp = sum(temps.values()) / len(temps)
        
        print(f"\n{'='*60}")
        print(f"RAID1 (md1) Thermal Status - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        for drive, temp in temps.items():
            status, emoji = self.get_status(temp)
            model = "Crucial T700 4TB"  # Both RAID1 drives are Crucial T700 4TB
            print(f"{drive} ({model:18}): {temp:>5.1f}°C {emoji} {status}")
        
        print(f"{'-'*60}")
        print(f"Average: {avg_temp:>5.1f}°C | Maximum: {max_temp:>5.1f}°C")
        
        # Thermal recommendations
        if max_temp >= self.thresholds['hot']:
            print(f"\n⚠️  WARNING: Drives running hot!")
            print(f"   Recommendations:")
            print(f"   - Increase I/O throttling")
            print(f"   - Add cooling time between operations")
            print(f"   - Check case ventilation")
        elif max_temp >= self.thresholds['warm']:
            print(f"\n⚠️  Drives are warming up - monitor closely")
        
        print(f"{'='*60}")
        
        return max_temp
    
    def monitor_continuous(self, interval: int = 5, alert_temp: float = 70, log_file: Optional[str] = None) -> None:
        """Continuously monitor temperatures with file logging."""
        import logging
        
        # Setup file logging if requested
        if log_file:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_file)
                ]
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Started RAID1 thermal monitoring (interval: {interval}s, alert: {alert_temp}°C)")
        else:
            logger = None
        
        print(f"Monitoring RAID1 temperatures every {interval}s")
        print(f"Alert threshold: {alert_temp}°C")
        if log_file:
            print(f"Logging to: {log_file}")
        print("Press Ctrl+C to stop\n")
        
        alert_count = 0
        
        try:
            while True:
                max_temp = self.print_status()
                
                if max_temp:
                    # Log current temperatures
                    temps = self.get_temperatures()
                    temp_str = ", ".join([f"{drive}: {temp:.1f}°C" for drive, temp in temps.items()])
                    
                    if logger:
                        if max_temp >= alert_temp:
                            alert_count += 1
                            logger.warning(f"THERMAL ALERT #{alert_count}: Max {max_temp:.1f}°C ({temp_str})")
                        elif max_temp >= alert_temp - 5:  # Log when approaching threshold
                            logger.info(f"THERMAL WARNING: Approaching threshold - {temp_str}")
                        else:
                            # Log every 10th reading when normal
                            if int(time.time()) % (interval * 10) < interval:
                                logger.info(f"THERMAL NORMAL: {temp_str}")
                    
                    if max_temp >= alert_temp:
                        # Sound alert (terminal bell)
                        print('\a', end='', flush=True)
                        if logger:
                            logger.critical(f"TEMPERATURE CRITICAL: {max_temp:.1f}°C - RECOMMEND IMMEDIATE ACTION")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            if logger:
                logger.info("RAID1 thermal monitoring stopped by user")
            print("\n\nMonitoring stopped.")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="RAID1 thermal monitor")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=5, help="Update interval (seconds)")
    parser.add_argument("--alert", type=int, default=70, help="Alert temperature (°C)")
    parser.add_argument("--log-file", type=str, help="Log file path (e.g., raid1_thermal.log)")
    
    args = parser.parse_args()
    
    monitor = RAID1Monitor()
    
    if args.watch:
        monitor.monitor_continuous(args.interval, args.alert, args.log_file)
    else:
        monitor.print_status()


if __name__ == "__main__":
    main()