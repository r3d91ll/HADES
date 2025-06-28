#!/usr/bin/env python3
"""
NVMe Thermal Monitor for RAID Arrays

Maps temperature sensors to physical NVMe drives and monitors thermal status
with special attention to RAID0 arrays used for ML processing.
"""

import subprocess
import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class NVMeDrive:
    """NVMe drive information."""
    device: str  # e.g., /dev/nvme0n1
    model: str
    serial: str
    pci_address: str
    temperature_c: float
    sensor_name: str  # e.g., nvme-pci-8100


@dataclass
class RAIDArray:
    """RAID array information."""
    md_device: str  # e.g., md0
    level: str  # e.g., raid0
    members: List[str]  # e.g., [nvme2n1, nvme3n1]
    mount_points: List[str]


class ThermalStatus(Enum):
    """Thermal status levels."""
    NORMAL = "normal"       # < 60°C
    WARM = "warm"          # 60-70°C
    HOT = "hot"            # 70-80°C
    CRITICAL = "critical"   # > 80°C


class NVMeThermalMonitor:
    """Monitor NVMe temperatures with RAID awareness."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.nvme_drives: Dict[str, NVMeDrive] = {}
        self.raid_arrays: Dict[str, RAIDArray] = {}
        self.sensor_to_device: Dict[str, str] = {}
        
    def discover_nvme_drives(self) -> Dict[str, NVMeDrive]:
        """Discover all NVMe drives and their properties."""
        drives = {}
        
        try:
            # Get NVMe list
            result = subprocess.run(['nvme', 'list', '-o', 'json'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for device in data.get('Devices', []):
                    dev_path = device.get('DevicePath', '')
                    if not dev_path:
                        continue
                    
                    # Get device name (e.g., nvme0n1)
                    dev_name = Path(dev_path).name
                    
                    # Get detailed info
                    id_result = subprocess.run(['nvme', 'id-ctrl', dev_path, '-o', 'json'],
                                             capture_output=True, text=True, timeout=5)
                    if id_result.returncode == 0:
                        id_data = json.loads(id_result.stdout)
                        model = id_data.get('mn', 'Unknown')
                        serial = id_data.get('sn', 'Unknown').strip()
                    else:
                        model = device.get('ModelNumber', 'Unknown')
                        serial = device.get('SerialNumber', 'Unknown')
                    
                    # Get PCI address
                    pci_addr = self._get_pci_address(dev_name)
                    
                    drives[dev_name] = NVMeDrive(
                        device=dev_path,
                        model=model,
                        serial=serial,
                        pci_address=pci_addr,
                        temperature_c=0.0,
                        sensor_name=""
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to discover NVMe drives: {e}")
        
        self.nvme_drives = drives
        return drives
    
    def _get_pci_address(self, dev_name: str) -> str:
        """Get PCI address for an NVMe device."""
        try:
            # Get the controller name (nvme0, nvme1, etc.) from device name
            ctrl_name = dev_name.replace('n1', '')  # nvme0n1 -> nvme0
            
            # Read the symlink to get PCI address
            ctrl_path = f"/sys/class/nvme/{ctrl_name}"
            if Path(ctrl_path).exists() and Path(ctrl_path).is_symlink():
                link_target = os.readlink(ctrl_path)
                # Extract PCI address from path like ../../devices/pci0000:80/0000:80:01.1/0000:81:00.0/nvme/nvme0
                pci_match = re.search(r'([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9])', link_target)
                if pci_match:
                    addr = pci_match.group(1)
                    self.logger.debug(f"{dev_name} PCI address: {addr}")
                    return addr
                    
        except Exception as e:
            self.logger.debug(f"Error getting PCI address for {dev_name}: {e}")
        
        return "Unknown"
    
    def map_sensors_to_drives(self) -> None:
        """Map temperature sensors to NVMe drives using PCI addresses."""
        try:
            # First, let's see what PCI addresses we have
            self.logger.info("Drive PCI addresses:")
            for dev_name, drive in self.nvme_drives.items():
                self.logger.info(f"  {dev_name}: {drive.pci_address}")
            
            # Parse sensors output
            result = subprocess.run(['sensors', '-j'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                sensor_data = json.loads(result.stdout)
                
                self.logger.info("Available NVMe sensors:")
                for sensor_name in sensor_data.keys():
                    if sensor_name.startswith('nvme-pci-'):
                        self.logger.info(f"  {sensor_name}")
                
                # Try to match sensors to drives
                for sensor_name, sensor_info in sensor_data.items():
                    if sensor_name.startswith('nvme-pci-'):
                        # Extract PCI bus from sensor name (e.g., "8100" -> "81:00")
                        pci_part = sensor_name.split('-')[2]  # e.g., "8100"
                        if len(pci_part) == 4:
                            pci_bus = f"{pci_part[:2]}:{pci_part[2:]}"  # "81:00"
                            
                            # Match with drive PCI addresses
                            for dev_name, drive in self.nvme_drives.items():
                                # Check if this bus matches the drive's PCI address
                                if pci_bus in drive.pci_address:
                                    drive.sensor_name = sensor_name
                                    self.sensor_to_device[sensor_name] = dev_name
                                    self.logger.info(f"Mapped {sensor_name} -> {dev_name} ({drive.model})")
                                    break
                
                # Log unmapped drives
                for dev_name, drive in self.nvme_drives.items():
                    if not drive.sensor_name:
                        self.logger.warning(f"No sensor mapped for {dev_name} ({drive.pci_address})")
                
        except Exception as e:
            self.logger.error(f"Failed to map sensors: {e}")
    
    def discover_raid_arrays(self) -> Dict[str, RAIDArray]:
        """Discover RAID arrays and their members."""
        arrays = {}
        
        try:
            # Parse /proc/mdstat
            with open('/proc/mdstat', 'r') as f:
                content = f.read()
            
            # Find all md devices
            md_pattern = r'(md\d+) : active (\w+) (.+)'
            for match in re.finditer(md_pattern, content):
                md_name = match.group(1)
                raid_level = match.group(2)
                members_str = match.group(3)
                
                # Extract member devices
                members = []
                member_pattern = r'(\w+)\[\d+\]'
                for member_match in re.finditer(member_pattern, members_str):
                    member = member_match.group(1)
                    # Convert to base device name (remove partition)
                    base_device = re.sub(r'p\d+$', '', member)
                    members.append(base_device)
                
                # Find mount points
                mount_points = self._find_mount_points(f"/dev/{md_name}")
                
                arrays[md_name] = RAIDArray(
                    md_device=md_name,
                    level=raid_level,
                    members=members,
                    mount_points=mount_points
                )
                
        except Exception as e:
            self.logger.error(f"Failed to discover RAID arrays: {e}")
        
        self.raid_arrays = arrays
        return arrays
    
    def _find_mount_points(self, device: str) -> List[str]:
        """Find mount points for a device."""
        mount_points = []
        
        try:
            # Check LVM on top of RAID
            lvm_result = subprocess.run(['pvs', '--noheadings', '-o', 'pv_name,vg_name'],
                                      capture_output=True, text=True)
            if lvm_result.returncode == 0:
                for line in lvm_result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == device:
                        vg_name = parts[1]
                        
                        # Get LVs in this VG
                        lv_result = subprocess.run(['lvs', '--noheadings', '-o', 'lv_name,vg_name'],
                                                 capture_output=True, text=True)
                        if lv_result.returncode == 0:
                            for lv_line in lv_result.stdout.strip().split('\n'):
                                lv_parts = lv_line.split()
                                if len(lv_parts) >= 2 and lv_parts[1] == vg_name:
                                    lv_device = f"/dev/mapper/{vg_name}-{lv_parts[0]}"
                                    
                                    # Find mount point
                                    with open('/proc/mounts', 'r') as f:
                                        for mount_line in f:
                                            mount_parts = mount_line.split()
                                            if mount_parts[0] == lv_device:
                                                mount_points.append(mount_parts[1])
            
            # Direct mount check
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts[0] == device:
                        mount_points.append(parts[1])
                        
        except Exception as e:
            self.logger.error(f"Failed to find mount points: {e}")
        
        return mount_points
    
    def get_current_temperatures(self) -> Dict[str, float]:
        """Get current temperatures for all NVMe drives."""
        temps = {}
        
        try:
            result = subprocess.run(['sensors', '-j'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                sensor_data = json.loads(result.stdout)
                
                for sensor_name, device in self.sensor_to_device.items():
                    if sensor_name in sensor_data:
                        sensor_info = sensor_data[sensor_name]
                        
                        # Look for Composite temperature
                        for key, value in sensor_info.items():
                            if 'Composite' in key and isinstance(value, dict):
                                temp_key = next((k for k in value.keys() if 'input' in k), None)
                                if temp_key:
                                    temp_c = float(value[temp_key])
                                    temps[device] = temp_c
                                    self.nvme_drives[device].temperature_c = temp_c
                                    break
                        
        except Exception as e:
            self.logger.error(f"Failed to get temperatures: {e}")
        
        return temps
    
    def get_raid_temperatures(self, raid_name: str) -> Dict[str, float]:
        """Get temperatures for all drives in a RAID array."""
        temps = {}
        
        if raid_name in self.raid_arrays:
            array = self.raid_arrays[raid_name]
            current_temps = self.get_current_temperatures()
            
            for member in array.members:
                if member in current_temps:
                    temps[member] = current_temps[member]
        
        return temps
    
    def get_thermal_status(self, temp_c: float) -> ThermalStatus:
        """Get thermal status for a temperature."""
        if temp_c >= 80:
            return ThermalStatus.CRITICAL
        elif temp_c >= 70:
            return ThermalStatus.HOT
        elif temp_c >= 60:
            return ThermalStatus.WARM
        else:
            return ThermalStatus.NORMAL
    
    def print_thermal_report(self) -> None:
        """Print a comprehensive thermal report."""
        print("\n" + "="*80)
        print("NVMe THERMAL REPORT")
        print("="*80)
        
        # Update temperatures
        self.get_current_temperatures()
        
        # Print individual drives
        print("\nINDIVIDUAL DRIVES:")
        print("-"*80)
        print(f"{'Device':<12} {'Model':<30} {'Temp':<8} {'Status':<10} {'Sensor':<20}")
        print("-"*80)
        
        for dev_name, drive in sorted(self.nvme_drives.items()):
            status = self.get_thermal_status(drive.temperature_c)
            status_emoji = {
                ThermalStatus.NORMAL: "✅",
                ThermalStatus.WARM: "⚠️",
                ThermalStatus.HOT: "🔥",
                ThermalStatus.CRITICAL: "🚨"
            }[status]
            
            print(f"{dev_name:<12} {drive.model[:30]:<30} "
                  f"{drive.temperature_c:>6.1f}°C {status_emoji} {status.value:<8} "
                  f"{drive.sensor_name:<20}")
        
        # Print RAID arrays
        print("\nRAID ARRAYS:")
        print("-"*80)
        
        for raid_name, array in sorted(self.raid_arrays.items()):
            print(f"\n{raid_name} ({array.level}):")
            temps = self.get_raid_temperatures(raid_name)
            
            if temps:
                max_temp = max(temps.values())
                avg_temp = sum(temps.values()) / len(temps)
                status = self.get_thermal_status(max_temp)
                
                print(f"  Members: {', '.join(array.members)}")
                print(f"  Max Temp: {max_temp:.1f}°C ({status.value})")
                print(f"  Avg Temp: {avg_temp:.1f}°C")
                print(f"  Mount Points:")
                for mount in array.mount_points[:3]:  # Show first 3
                    print(f"    - {mount}")
                if len(array.mount_points) > 3:
                    print(f"    ... and {len(array.mount_points) - 3} more")
        
        print("\n" + "="*80)


def main() -> None:
    """Main entry point for thermal monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NVMe thermal monitor for RAID arrays")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor temperatures")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")
    parser.add_argument("--raid", type=str, help="Monitor specific RAID array (e.g., md0)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    monitor = NVMeThermalMonitor()
    
    print("Discovering NVMe drives...")
    monitor.discover_nvme_drives()
    
    print("Mapping temperature sensors...")
    monitor.map_sensors_to_drives()
    
    print("Discovering RAID arrays...")
    monitor.discover_raid_arrays()
    
    if args.watch:
        print(f"\nMonitoring temperatures every {args.interval} seconds (Ctrl+C to stop)...")
        try:
            while True:
                monitor.print_thermal_report()
                
                if args.raid:
                    # Focus on specific RAID
                    temps = monitor.get_raid_temperatures(args.raid)
                    if temps:
                        max_temp = max(temps.values())
                        if max_temp > 70:
                            print(f"\n⚠️  WARNING: {args.raid} running hot! Max: {max_temp:.1f}°C")
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        monitor.print_thermal_report()


if __name__ == "__main__":
    main()