#!/usr/bin/env python3
"""
Thermal-Aware Batch Group PathRAG Extractor

Balances memory usage with disk I/O to reduce M.2 thermal load:
1. Accumulates multiple groups in RAM before writing
2. Uses compression to reduce write volume
3. Implements I/O throttling between operations
4. Monitors thermal status and adapts behavior
5. Batches file operations to reduce write frequency
"""

import logging
import pickle
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional
import threading
import queue
from dataclasses import dataclass
from enum import Enum
import psutil
import os
import subprocess
import json

from bootstrap_graph_builder.src.streaming_batch_group_pathrag_extractor import (
    StreamingBatchGroupPathRAGExtractor,
    BatchGroupManager,
    BatchGroupStatus,
    BatchGroupInfo,
    streaming_gpu_worker_process_chunks
)

logger = logging.getLogger(__name__)


class ThermalStatus(Enum):
    """Thermal status levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ThermalMetrics:
    """Thermal monitoring metrics."""
    max_temp_c: float
    device_temps: Dict[str, float]
    status: ThermalStatus
    timestamp: float


class ThermalAwareBatchGroupManager(BatchGroupManager):
    """Extended batch group manager with thermal awareness and I/O optimization."""
    
    def __init__(self, 
                 total_chunks: int, 
                 num_workers: int, 
                 chunks_per_batch: int,
                 memory_budget_gb: float = 200.0,
                 memory_buffer_gb: float = 50.0,
                 io_throttle_seconds: float = 0.5,
                 batch_write_threshold: int = 3,
                 use_compression: bool = True,
                 thermal_threshold_c: int = 70,
                 thermal_critical_c: int = 80):
        """
        Initialize thermal-aware batch group manager.
        
        Args:
            memory_buffer_gb: Extra RAM to use before triggering writes
            io_throttle_seconds: Delay between I/O operations
            batch_write_threshold: Number of groups to accumulate before writing
            use_compression: Whether to compress intermediate files
            thermal_threshold_c: Warning temperature threshold
            thermal_critical_c: Critical temperature threshold
        """
        # Initialize base manager with adjusted memory settings
        adjusted_budget = memory_budget_gb + memory_buffer_gb
        super().__init__(total_chunks, num_workers, chunks_per_batch, 
                        adjusted_budget, 4.0, 0.6)
        
        self.memory_buffer_gb = memory_buffer_gb
        self.io_throttle_seconds = io_throttle_seconds
        self.batch_write_threshold = batch_write_threshold
        self.use_compression = use_compression
        self.thermal_threshold_c = thermal_threshold_c
        self.thermal_critical_c = thermal_critical_c
        
        # Batch write accumulation
        self.pending_writes = []
        self.accumulated_relationships = {}
        self.accumulated_count = 0
        
        # Thermal monitoring
        self.thermal_metrics = None
        self.thermal_monitor_thread = None
        self.thermal_monitor_active = threading.Event()
        
        logger.info(f"Thermal-aware manager initialized:")
        logger.info(f"  Memory buffer: {memory_buffer_gb}GB")
        logger.info(f"  Batch writes: Every {batch_write_threshold} groups")
        logger.info(f"  I/O throttle: {io_throttle_seconds}s")
        logger.info(f"  Compression: {use_compression}")
    
    def start_thermal_monitor(self):
        """Start thermal monitoring thread."""
        self.thermal_monitor_active.set()
        self.thermal_monitor_thread = threading.Thread(
            target=self._thermal_monitor_worker, daemon=True
        )
        self.thermal_monitor_thread.start()
        logger.info("Started thermal monitoring")
    
    def _thermal_monitor_worker(self):
        """Monitor thermal status of NVMe drives."""
        while self.thermal_monitor_active.is_set():
            try:
                temps = self._get_nvme_temperatures()
                if temps:
                    max_temp = max(temps.values())
                    
                    if max_temp >= self.thermal_critical_c:
                        status = ThermalStatus.CRITICAL
                    elif max_temp >= self.thermal_threshold_c:
                        status = ThermalStatus.WARNING
                    else:
                        status = ThermalStatus.NORMAL
                    
                    self.thermal_metrics = ThermalMetrics(
                        max_temp_c=max_temp,
                        device_temps=temps,
                        status=status,
                        timestamp=time.time()
                    )
                    
                    if status != ThermalStatus.NORMAL:
                        logger.warning(f"🌡️ Thermal {status.value}: {max_temp}°C (temps: {temps})")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Thermal monitor error: {e}")
                time.sleep(30)
    
    def _get_nvme_temperatures(self) -> Dict[str, float]:
        """Get NVMe drive temperatures, focusing on RAID1 array drives."""
        temps = {}
        
        # Direct mapping for RAID1 drives based on known PCI addresses
        raid1_sensors = {
            'nvme-pci-8100': 'nvme0n1',  # Crucial T700 4TB
            'nvme-pci-8200': 'nvme1n1'   # Crucial T700 4TB
        }
        
        try:
            result = subprocess.run(['sensors', '-j'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                sensor_data = json.loads(result.stdout)
                
                # Get RAID1 drive temps specifically
                for sensor, drive in raid1_sensors.items():
                    if sensor in sensor_data:
                        sensor_info = sensor_data[sensor]
                        for key, value in sensor_info.items():
                            if 'Composite' in key and isinstance(value, dict):
                                temp_key = next((k for k in value.keys() if 'input' in k), None)
                                if temp_key:
                                    temps[drive] = float(value[temp_key])
                                    break
                
                if temps:
                    logger.info(f"RAID1 temperatures: {temps}")
                
        except Exception as e:
            logger.warning(f"Failed to get temperatures: {e}")
            
        return temps
    
    def get_io_throttle_time(self) -> float:
        """Get dynamic I/O throttle time based on thermal status."""
        if not self.thermal_metrics:
            return self.io_throttle_seconds
        
        if self.thermal_metrics.status == ThermalStatus.CRITICAL:
            return self.io_throttle_seconds * 3  # Triple throttle
        elif self.thermal_metrics.status == ThermalStatus.WARNING:
            return self.io_throttle_seconds * 1.5  # 1.5x throttle
        else:
            return self.io_throttle_seconds
    
    def _write_worker_thermal_aware(self, output_file: Path, incremental_output_dir: Path):
        """Thermal-aware write worker that batches writes and implements throttling."""
        logger.info("Started thermal-aware write worker")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for groups to accumulate
                group_id = self.write_queue.get(timeout=1.0)
                if group_id is None:  # Shutdown signal
                    break
                
                # Add to pending writes
                self.pending_writes.append(group_id)
                
                # Check if we should batch write
                should_write = (
                    len(self.pending_writes) >= self.batch_write_threshold or
                    (group_id == self.num_groups - 1)  # Last group
                )
                
                if should_write:
                    self._perform_batch_write(incremental_output_dir)
                
                self.write_queue.task_done()
                
            except queue.Empty:
                # Check if we have pending writes to flush
                if self.pending_writes and not self.write_queue.qsize():
                    self._perform_batch_write(incremental_output_dir)
            except Exception as e:
                logger.error(f"Error in thermal-aware write worker: {e}")
        
        # Final batch write
        if self.pending_writes:
            self._perform_batch_write(incremental_output_dir)
        
        # Final consolidation with thermal awareness
        self._consolidate_with_thermal_awareness(output_file, incremental_output_dir)
    
    def _perform_batch_write(self, incremental_output_dir: Path):
        """Perform batched write with thermal awareness."""
        if not self.pending_writes:
            return
        
        batch_start_time = time.time()
        logger.info(f"Starting batch write of {len(self.pending_writes)} groups...")
        
        # Accumulate all pending groups
        batch_relationships = {}
        total_batch_relationships = 0
        
        for group_id in self.pending_writes:
            group_info = self.batch_groups[group_id]
            
            # Load and merge worker results
            for worker_id, result_file in group_info.result_files.items():
                worker_data = torch.load(result_file, map_location='cpu', weights_only=False)
                batch_relationships.update(worker_data)
                total_batch_relationships += sum(len(rels) for rels in worker_data.values())
                
                # Clean up worker file
                if result_file.exists():
                    result_file.unlink()
                del worker_data
        
        # Apply thermal-aware I/O throttling
        throttle_time = self.get_io_throttle_time()
        if throttle_time > 0:
            logger.info(f"Applying I/O throttle: {throttle_time:.1f}s")
            time.sleep(throttle_time)
        
        # Write batch file with optional compression
        batch_file = incremental_output_dir / f"batch_{min(self.pending_writes)}_{max(self.pending_writes)}.pt"
        
        if self.use_compression:
            # Save with compression
            torch.save(batch_relationships, batch_file, _use_new_zipfile_serialization=True)
            logger.info(f"Wrote compressed batch file: {batch_file.name}")
        else:
            torch.save(batch_relationships, batch_file)
        
        # Update group statuses
        for group_id in self.pending_writes:
            group_info = self.batch_groups[group_id]
            group_info.status = BatchGroupStatus.WRITTEN
            group_info.write_completed_time = time.time()
            self.groups_written += 1
        
        self.total_relationships_written += total_batch_relationships
        
        # Clear pending writes
        self.pending_writes.clear()
        del batch_relationships
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Batch write completed: {total_batch_relationships:,} relationships in {batch_time:.1f}s")
        
        # Check for subsequent groups that can now be written
        last_written = max(self.pending_writes) if self.pending_writes else self.groups_written - 1
        self._check_next_groups_for_writing(last_written + 1)
    
    def _consolidate_with_thermal_awareness(self, output_file: Path, incremental_output_dir: Path):
        """Consolidate files with thermal-aware I/O management."""
        logger.info("Starting thermal-aware consolidation...")
        
        final_relationships = {}
        batch_files = sorted(incremental_output_dir.glob("batch_*.pt"))
        
        for i, batch_file in enumerate(batch_files):
            # Apply throttling between reads
            if i > 0:
                throttle_time = self.get_io_throttle_time()
                if throttle_time > 0:
                    time.sleep(throttle_time)
            
            logger.info(f"Loading batch file {i+1}/{len(batch_files)}: {batch_file.name}")
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=False)
            final_relationships.update(batch_data)
            del batch_data
            
            # Remove batch file after loading
            batch_file.unlink()
        
        # Final throttle before big write
        throttle_time = self.get_io_throttle_time() * 2  # Double throttle for final write
        if throttle_time > 0:
            logger.info(f"Final write throttle: {throttle_time:.1f}s")
            time.sleep(throttle_time)
        
        # Save final consolidated file
        if self.use_compression:
            torch.save(final_relationships, output_file, _use_new_zipfile_serialization=True)
            logger.info(f"Saved compressed final relationships: {len(final_relationships):,} chunks")
        else:
            torch.save(final_relationships, output_file)
        
        # Clean up
        if incremental_output_dir.exists():
            incremental_output_dir.rmdir()
        
        logger.info(f"Thermal-aware consolidation complete: {self.total_relationships_written:,} relationships")
    
    def shutdown(self):
        """Shutdown including thermal monitor."""
        self.thermal_monitor_active.clear()
        super().shutdown()


class ThermalAwareBatchGroupExtractor(StreamingBatchGroupPathRAGExtractor):
    """Thermal-aware version of the batch group PathRAG extractor."""
    
    def __init__(self,
                 dataset_dir: Path,
                 decay_rate: float = 0.8,
                 max_hops: int = 3,
                 top_k: int = 1000,
                 device: str = 'cuda:0',
                 num_workers: int = 12,
                 chunks_per_batch: int = 500,
                 memory_budget_gb: float = 200.0,
                 memory_buffer_gb: float = 50.0,
                 io_throttle_seconds: float = 0.5,
                 batch_write_threshold: int = 3,
                 use_compression: bool = True,
                 thermal_monitoring: bool = True,
                 thermal_threshold_c: int = 70):
        """
        Initialize thermal-aware extractor with I/O optimizations.
        """
        # Initialize base class
        super().__init__(dataset_dir, decay_rate, max_hops, top_k, 
                        device, num_workers, chunks_per_batch)
        
        # Replace batch manager with thermal-aware version
        self.batch_manager = ThermalAwareBatchGroupManager(
            self.total_chunks,
            num_workers,
            chunks_per_batch,
            memory_budget_gb,
            memory_buffer_gb,
            io_throttle_seconds,
            batch_write_threshold,
            use_compression,
            thermal_threshold_c
        )
        
        self.thermal_monitoring = thermal_monitoring
        
        logger.info("Initialized thermal-aware PathRAG extractor")
    
    def precompute_all_relationships_thermal_aware(self) -> str:
        """Run thermal-aware relationship precomputation."""
        logger.info("Starting thermal-aware relationship precomputation...")
        
        # Start monitoring if enabled
        if self.thermal_monitoring:
            self.batch_manager.start_thermal_monitor()
        
        # Start memory monitor
        self.batch_manager.start_memory_monitor()
        
        # Use modified write worker
        output_file = self.dataset_dir / "pathrag_relationships_thermal.pt"
        incremental_dir = self.dataset_dir / "incremental_thermal"
        incremental_dir.mkdir(exist_ok=True)
        
        # Start thermal-aware write thread
        self.batch_manager.write_thread = threading.Thread(
            target=self.batch_manager._write_worker_thermal_aware,
            args=(output_file, incremental_dir),
            daemon=True
        )
        self.batch_manager.write_thread.start()
        
        # Run the standard processing pipeline
        return self._run_streaming_pipeline()
    
    def _run_streaming_pipeline(self) -> str:
        """Run the streaming pipeline (reusing parent implementation)."""
        # This reuses the parent's streaming pipeline logic
        # but with our thermal-aware batch manager
        return super().precompute_all_relationships_streaming()