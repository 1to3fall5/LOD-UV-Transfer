"""
Batch Processor - Handles batch UV transfer operations.
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import glob

from ..utils.logger import get_logger, OperationContext
from ..utils.error_handler import TransferError
from ..core.transfer_engine import UVTransferEngine, TransferConfig, TransferResult
from .config_manager import ConfigManager


@dataclass
class BatchItem:
    """Single batch transfer item."""
    source_file: str
    target_file: str
    output_file: str
    source_mesh: Optional[str] = None
    target_mesh: Optional[str] = None
    source_uv_channel: int = 0
    target_uv_channel: int = 0
    config: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    """Result of a batch operation."""
    total_items: int
    successful: int
    failed: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.successful / self.total_items


class BatchProcessor:
    """
    Handles batch UV transfer operations.
    
    Supports:
    - Multiple file pairs
    - LOD chain processing
    - Configuration templates
    - Progress tracking
    """
    
    def __init__(
        self,
        engine: Optional[UVTransferEngine] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        self.logger = get_logger("uv_transfer.batch")
        self.engine = engine or UVTransferEngine()
        self.config_manager = config_manager or ConfigManager()
    
    def process_batch(
        self,
        items: List[BatchItem],
        default_config: Optional[TransferConfig] = None,
        stop_on_error: bool = False
    ) -> BatchResult:
        """
        Process a batch of transfer items.
        
        Args:
            items: List of BatchItem objects
            default_config: Default configuration for all items
            stop_on_error: Whether to stop on first error
        
        Returns:
            BatchResult with all results
        """
        result = BatchResult(
            total_items=len(items),
            successful=0,
            failed=0
        )
        
        self.logger.info(f"Starting batch processing: {len(items)} items")
        
        for i, item in enumerate(items):
            self.logger.info(f"Processing item {i+1}/{len(items)}: {item.source_file}")
            
            try:
                transfer_result = self._process_item(item, default_config)
                
                result.results.append({
                    'item': asdict(item),
                    'success': transfer_result.success,
                    'details': {
                        'source_vertices': transfer_result.source_vertices,
                        'target_vertices': transfer_result.target_vertices,
                        'matched_vertices': transfer_result.matched_vertices,
                        'accuracy': transfer_result.accuracy,
                        'match_rate': transfer_result.match_rate,
                    }
                })
                
                if transfer_result.success:
                    result.successful += 1
                else:
                    result.failed += 1
                
            except Exception as e:
                self.logger.error(f"Item {i+1} failed: {e}")
                result.failed += 1
                result.results.append({
                    'item': asdict(item),
                    'success': False,
                    'error': str(e)
                })
                
                if stop_on_error:
                    break
        
        self.logger.info(
            f"Batch complete: {result.successful} successful, "
            f"{result.failed} failed"
        )
        
        return result
    
    def _process_item(
        self,
        item: BatchItem,
        default_config: Optional[TransferConfig]
    ) -> TransferResult:
        """Process a single batch item."""
        config = default_config or TransferConfig()
        
        if item.config:
            config = TransferConfig(**{**asdict(config), **item.config})
        
        config.source_uv_channel = item.source_uv_channel
        config.target_uv_channel = item.target_uv_channel
        
        self.engine.load_source(item.source_file)
        self.engine.load_target(item.target_file)
        
        result = self.engine.transfer(
            source_mesh_name=item.source_mesh,
            target_mesh_name=item.target_mesh,
            config=config
        )
        
        if result.success:
            self.engine.save_result(item.output_file)
        
        return result
    
    def process_lod_chain(
        self,
        lod_files: List[str],
        output_dir: str,
        source_uv_channel: int = 0,
        target_uv_channel: int = 0,
        config: Optional[TransferConfig] = None
    ) -> BatchResult:
        """
        Process a chain of LOD files.
        
        Transfers UV from LOD0 to all subsequent LODs.
        
        Args:
            lod_files: List of LOD file paths (sorted by LOD level)
            output_dir: Output directory for processed files
            source_uv_channel: Source UV channel
            target_uv_channel: Target UV channel
            config: Transfer configuration
        
        Returns:
            BatchResult with all results
        """
        if len(lod_files) < 2:
            raise TransferError("Need at least 2 LOD files for chain processing")
        
        os.makedirs(output_dir, exist_ok=True)
        
        items = []
        source_file = lod_files[0]
        
        for i, target_file in enumerate(lod_files[1:], start=1):
            output_file = os.path.join(
                output_dir,
                os.path.basename(target_file)
            )
            
            items.append(BatchItem(
                source_file=source_file,
                target_file=target_file,
                output_file=output_file,
                source_uv_channel=source_uv_channel,
                target_uv_channel=target_uv_channel
            ))
        
        return self.process_batch(items, config)
    
    def auto_detect_lod_files(
        self,
        directory: str,
        pattern: str = "*.fbx"
    ) -> List[str]:
        """
        Auto-detect LOD files in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
        
        Returns:
            List of LOD file paths sorted by LOD level
        """
        files = glob.glob(os.path.join(directory, pattern))
        
        def extract_lod_level(filepath):
            name = os.path.basename(filepath)
            import re
            match = re.search(r'LOD(\d+)', name, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 999
        
        files.sort(key=extract_lod_level)
        
        self.logger.info(f"Detected {len(files)} LOD files")
        for f in files:
            self.logger.debug(f"  {os.path.basename(f)}")
        
        return files
    
    def create_batch_from_directory(
        self,
        source_dir: str,
        target_dir: str,
        output_dir: str,
        source_uv_channel: int = 0,
        target_uv_channel: int = 0,
        pattern: str = "*.fbx"
    ) -> List[BatchItem]:
        """
        Create batch items from directory structure.
        
        Args:
            source_dir: Source files directory
            target_dir: Target files directory
            output_dir: Output directory
            source_uv_channel: Source UV channel
            target_uv_channel: Target UV channel
            pattern: File pattern to match
        
        Returns:
            List of BatchItem objects
        """
        source_files = sorted(glob.glob(os.path.join(source_dir, pattern)))
        target_files = sorted(glob.glob(os.path.join(target_dir, pattern)))
        
        if len(source_files) != len(target_files):
            self.logger.warning(
                f"Source ({len(source_files)}) and target ({len(target_files)}) "
                f"file counts don't match"
            )
        
        items = []
        for src, tgt in zip(source_files, target_files):
            output_file = os.path.join(output_dir, os.path.basename(tgt))
            
            items.append(BatchItem(
                source_file=src,
                target_file=tgt,
                output_file=output_file,
                source_uv_channel=source_uv_channel,
                target_uv_channel=target_uv_channel
            ))
        
        return items
    
    def save_batch_config(
        self,
        items: List[BatchItem],
        filepath: str
    ):
        """Save batch configuration to file."""
        data = {
            'version': '1.0',
            'items': [asdict(item) for item in items]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved batch config to {filepath}")
    
    def load_batch_config(self, filepath: str) -> List[BatchItem]:
        """Load batch configuration from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        for item_data in data.get('items', []):
            items.append(BatchItem(**item_data))
        
        self.logger.info(f"Loaded {len(items)} items from {filepath}")
        return items
    
    def generate_report(
        self,
        result: BatchResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a batch processing report.
        
        Args:
            result: BatchResult to report
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        lines = [
            "Batch Processing Report",
            "=" * 50,
            f"Total Items: {result.total_items}",
            f"Successful: {result.successful}",
            f"Failed: {result.failed}",
            f"Success Rate: {result.success_rate:.1%}",
            "",
            "Item Details:",
            "-" * 50,
        ]
        
        for i, item_result in enumerate(result.results, 1):
            item = item_result.get('item', {})
            success = item_result.get('success', False)
            
            status = "OK" if success else "FAILED"
            lines.append(f"\n[{i}] {status}")
            lines.append(f"    Source: {item.get('source_file', 'N/A')}")
            lines.append(f"    Target: {item.get('target_file', 'N/A')}")
            lines.append(f"    Output: {item.get('output_file', 'N/A')}")
            
            if success and 'details' in item_result:
                details = item_result['details']
                lines.append(f"    Vertices: {details.get('target_vertices', 0)}")
                lines.append(f"    Match Rate: {details.get('match_rate', 0):.1%}")
                lines.append(f"    Accuracy: {details.get('accuracy', 0):.6f}")
            
            if not success and 'error' in item_result:
                lines.append(f"    Error: {item_result['error']}")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
