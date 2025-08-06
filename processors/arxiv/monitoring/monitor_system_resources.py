#!/usr/bin/env python3
"""
Monitor system resources during processing.
"""

import psutil
import subprocess
import time
from datetime import datetime

def get_process_info():
    """Get info about batch processing processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
        try:
            if 'python' in proc.info['name'] and any(x in ' '.join(proc.cmdline()) for x in ['batch_preprocess', 'enhanced']):
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu': proc.cpu_percent(interval=0.1),
                    'mem': proc.memory_info().rss / 1024 / 1024 / 1024,  # GB
                    'threads': proc.info['num_threads']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def monitor_resources():
    """Monitor system resources"""
    
    print(f"=== SYSTEM RESOURCE MONITOR ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System memory
    mem = psutil.virtual_memory()
    print("=== MEMORY USAGE ===")
    print(f"Total: {mem.total / 1024**3:.1f} GB")
    print(f"Used: {mem.used / 1024**3:.1f} GB ({mem.percent}%)")
    print(f"Available: {mem.available / 1024**3:.1f} GB")
    print(f"Cached: {mem.cached / 1024**3:.1f} GB")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    print("\n=== CPU USAGE ===")
    print(f"Overall: {sum(cpu_percent)/len(cpu_percent):.1f}%")
    print(f"Cores in use: {sum(1 for x in cpu_percent if x > 10)}/{len(cpu_percent)}")
    
    # GPU status
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\n=== GPU STATUS ===")
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                gpu_id = parts[0]
                gpu_name = parts[1]
                util = parts[2]
                mem_used = int(parts[3])
                mem_total = int(parts[4])
                temp = parts[5]
                print(f"GPU {gpu_id}: {util}% util, {mem_used}/{mem_total} MB ({mem_used/mem_total*100:.1f}%), {temp}°C")
    except:
        pass
    
    # Process information
    processes = get_process_info()
    if processes:
        print("\n=== BATCH PROCESS INFO ===")
        total_threads = 0
        total_mem = 0
        for proc in processes:
            print(f"PID {proc['pid']}: {proc['cpu']:.1f}% CPU, {proc['mem']:.1f} GB RAM, {proc['threads']} threads")
            total_threads += proc['threads']
            total_mem += proc['mem']
        print(f"\nTotal threads: {total_threads}")
        print(f"Total memory: {total_mem:.1f} GB")
    
    # Check for high memory pressure
    if mem.percent > 90:
        print("\n⚠️  WARNING: Memory usage above 90%!")
    if mem.available < 10 * 1024**3:  # Less than 10GB available
        print("\n⚠️  WARNING: Less than 10GB memory available!")

def continuous_monitor(interval=30):
    """Continuously monitor resources"""
    print("Starting continuous monitoring (Ctrl+C to stop)")
    print(f"Update interval: {interval} seconds")
    print("-" * 80)
    
    try:
        while True:
            monitor_resources()
            print("\n" + "-" * 80 + "\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        continuous_monitor(interval)
    else:
        monitor_resources()