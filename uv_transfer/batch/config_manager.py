"""
Configuration Manager - Manages transfer configurations.
"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.error_handler import ConfigError
from ..core.transfer_engine import TransferConfig, TransferMode


@dataclass
class PresetConfig:
    """A named configuration preset."""
    name: str
    description: str
    config: TransferConfig


class ConfigManager:
    """
    Manages transfer configurations and presets.
    
    Features:
    - Save/load configurations
    - Preset management
    - Configuration validation
    """
    
    DEFAULT_PRESETS = [
        PresetConfig(
            name="default",
            description="Default configuration for general use",
            config=TransferConfig(
                mode=TransferMode.SPATIAL,
                validate_source=True,
                validate_result=True
            )
        ),
        PresetConfig(
            name="high_quality",
            description="High quality transfer with strict validation",
            config=TransferConfig(
                mode=TransferMode.INTERPOLATED,
                validate_source=True,
                validate_result=True,
                accuracy_threshold=0.0001,
                smooth_iterations=2
            )
        ),
        PresetConfig(
            name="fast",
            description="Fast transfer with minimal validation",
            config=TransferConfig(
                mode=TransferMode.DIRECT,
                validate_source=False,
                validate_result=False
            )
        ),
        PresetConfig(
            name="lod_optimized",
            description="Optimized for LOD model chains",
            config=TransferConfig(
                mode=TransferMode.SPATIAL,
                validate_source=True,
                validate_result=True,
                vertex_match_threshold=0.005,
                preserve_boundary=True
            )
        ),
    ]
    
    def __init__(self, config_dir: Optional[str] = None):
        self.logger = get_logger("uv_transfer.config")
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".uv_transfer"
        self.presets: Dict[str, PresetConfig] = {}
        
        self._load_default_presets()
        self._load_user_presets()
    
    def _load_default_presets(self):
        """Load default presets."""
        for preset in self.DEFAULT_PRESETS:
            self.presets[preset.name] = preset
    
    def _load_user_presets(self):
        """Load user-defined presets from config directory."""
        if not self.config_dir.exists():
            return
        
        preset_file = self.config_dir / "presets.json"
        if preset_file.exists():
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for preset_data in data.get('presets', []):
                    config_dict = preset_data.get('config', {})
                    config_dict['mode'] = TransferMode(config_dict.get('mode', 'spatial'))
                    
                    preset = PresetConfig(
                        name=preset_data['name'],
                        description=preset_data.get('description', ''),
                        config=TransferConfig(**config_dict)
                    )
                    self.presets[preset.name] = preset
                
                self.logger.info(f"Loaded {len(data.get('presets', []))} user presets")
            except Exception as e:
                self.logger.warning(f"Failed to load user presets: {e}")
    
    def get_preset(self, name: str) -> Optional[TransferConfig]:
        """Get configuration by preset name."""
        preset = self.presets.get(name)
        return preset.config if preset else None
    
    def get_preset_names(self) -> List[str]:
        """Get list of available preset names."""
        return list(self.presets.keys())
    
    def get_preset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get preset information."""
        preset = self.presets.get(name)
        if preset:
            return {
                'name': preset.name,
                'description': preset.description,
                'config': asdict(preset.config)
            }
        return None
    
    def save_preset(
        self,
        name: str,
        config: TransferConfig,
        description: str = ""
    ):
        """Save a configuration as a preset."""
        preset = PresetConfig(
            name=name,
            description=description,
            config=config
        )
        
        self.presets[name] = preset
        self._save_user_presets()
        
        self.logger.info(f"Saved preset: {name}")
    
    def _save_user_presets(self):
        """Save user presets to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        preset_file = self.config_dir / "presets.json"
        
        user_presets = []
        for name, preset in self.presets.items():
            if name not in [p.name for p in self.DEFAULT_PRESETS]:
                user_presets.append({
                    'name': preset.name,
                    'description': preset.description,
                    'config': asdict(preset.config)
                })
        
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump({'presets': user_presets}, f, indent=2)
    
    def delete_preset(self, name: str) -> bool:
        """Delete a user preset."""
        if name in [p.name for p in self.DEFAULT_PRESETS]:
            self.logger.warning(f"Cannot delete default preset: {name}")
            return False
        
        if name in self.presets:
            del self.presets[name]
            self._save_user_presets()
            self.logger.info(f"Deleted preset: {name}")
            return True
        
        return False
    
    def validate_config(self, config: TransferConfig) -> List[str]:
        """Validate a configuration and return any issues."""
        issues = []
        
        if config.accuracy_threshold <= 0:
            issues.append("accuracy_threshold must be positive")
        
        if config.vertex_match_threshold <= 0:
            issues.append("vertex_match_threshold must be positive")
        
        if config.max_iterations <= 0:
            issues.append("max_iterations must be positive")
        
        if config.smooth_iterations < 0:
            issues.append("smooth_iterations cannot be negative")
        
        if not isinstance(config.mode, TransferMode):
            issues.append(f"Invalid mode: {config.mode}")
        
        return issues
    
    def create_config(
        self,
        mode: str = "spatial",
        source_uv_channel: int = 0,
        target_uv_channel: int = 0,
        **kwargs
    ) -> TransferConfig:
        """Create a configuration with specified parameters."""
        mode_enum = TransferMode(mode.lower())
        
        return TransferConfig(
            mode=mode_enum,
            source_uv_channel=source_uv_channel,
            target_uv_channel=target_uv_channel,
            **kwargs
        )
    
    def export_config(self, config: TransferConfig, filepath: str):
        """Export configuration to file."""
        data = asdict(config)
        data['mode'] = data['mode'].value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Exported config to {filepath}")
    
    def import_config(self, filepath: str) -> TransferConfig:
        """Import configuration from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['mode'] = TransferMode(data.get('mode', 'spatial'))
        
        config = TransferConfig(**data)
        
        issues = self.validate_config(config)
        if issues:
            raise ConfigError(
                f"Invalid configuration: {', '.join(issues)}",
                error_code=4003
            )
        
        self.logger.info(f"Imported config from {filepath}")
        return config
