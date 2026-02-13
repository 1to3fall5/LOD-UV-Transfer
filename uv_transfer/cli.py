"""
Command Line Interface for UV Transfer Tool.
"""

import argparse
import sys
import os
import json
from typing import Optional

from .utils.logger import setup_logger, get_logger
from .core.transfer_engine import UVTransferEngine, TransferConfig, TransferMode, TransferAlgorithm
from .batch.batch_processor import BatchProcessor, BatchItem
from .batch.config_manager import ConfigManager
from .visualization.uv_viewer import UVViewer
from .visualization.comparison import UVComparison
from .fbx.fbx_handler import FBXHandler


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="UV Transfer Tool - Transfer UV channels between LOD models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transfer UV from LOD0 to LOD1
  uv-transfer transfer source.fbx target.fbx -o output.fbx
  
  # Transfer specific UV channel
  uv-transfer transfer source.fbx target.fbx --source-uv 1 --target-uv 1
  
  # Process LOD chain
  uv-transfer lod-chain ./models ./output
  
  # View UV layout
  uv-transfer view model.fbx --uv-channel 0
  
  # Validate UV data
  uv-transfer validate model.fbx
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    transfer_parser = subparsers.add_parser('transfer', help='Transfer UV between models')
    transfer_parser.add_argument('source', help='Source FBX file')
    transfer_parser.add_argument('target', help='Target FBX file')
    transfer_parser.add_argument('-o', '--output', required=True, help='Output file path')
    transfer_parser.add_argument('--source-uv', type=int, default=0, help='Source UV channel index')
    transfer_parser.add_argument('--target-uv', type=int, default=0, help='Target UV channel index')
    transfer_parser.add_argument('--source-mesh', help='Source mesh name (auto-detect if not specified)')
    transfer_parser.add_argument('--target-mesh', help='Target mesh name (auto-detect if not specified)')
    transfer_parser.add_argument('--mode', choices=['direct', 'spatial', 'interpolated'], 
                                 default='spatial', help='Transfer mode')
    transfer_parser.add_argument('--algorithm', choices=['triangle_center', 'area_weighted', 'normal_aware'], 
                                 default='triangle_center', 
                                 help='UV transfer algorithm (triangle_center: best balance, area_weighted: preserves UV islands better, normal_aware: best for curved surfaces)')
    transfer_parser.add_argument('--preset', help='Use configuration preset')
    transfer_parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    transfer_parser.add_argument('--smooth', type=int, default=0, help='Smooth iterations')
    transfer_parser.add_argument('--threshold', type=float, default=0.01, help='Vertex match threshold')
    
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple files')
    batch_parser.add_argument('config', help='Batch configuration JSON file')
    batch_parser.add_argument('-o', '--output', help='Output report file')
    batch_parser.add_argument('--stop-on-error', action='store_true', help='Stop on first error')
    
    lod_parser = subparsers.add_parser('lod-chain', help='Process LOD chain')
    lod_parser.add_argument('directory', help='Directory containing LOD files')
    lod_parser.add_argument('output', help='Output directory')
    lod_parser.add_argument('--source-uv', type=int, default=0, help='Source UV channel')
    lod_parser.add_argument('--target-uv', type=int, default=0, help='Target UV channel')
    
    view_parser = subparsers.add_parser('view', help='Visualize UV layout')
    view_parser.add_argument('file', help='FBX file to view')
    view_parser.add_argument('--uv-channel', type=int, default=0, help='UV channel to view')
    view_parser.add_argument('--output', help='Output image file')
    view_parser.add_argument('--show-islands', action='store_true', help='Show UV islands')
    
    validate_parser = subparsers.add_parser('validate', help='Validate UV data')
    validate_parser.add_argument('file', help='FBX file to validate')
    validate_parser.add_argument('--uv-channel', type=int, default=0, help='UV channel to validate')
    validate_parser.add_argument('--output', help='Output report file')
    
    compare_parser = subparsers.add_parser('compare', help='Compare UV between models')
    compare_parser.add_argument('source', help='Source FBX file')
    compare_parser.add_argument('target', help='Target FBX file')
    compare_parser.add_argument('--source-uv', type=int, default=0, help='Source UV channel')
    compare_parser.add_argument('--target-uv', type=int, default=0, help='Target UV channel')
    compare_parser.add_argument('--output-dir', help='Output directory for visualizations')
    
    info_parser = subparsers.add_parser('info', help='Show FBX file info')
    info_parser.add_argument('file', help='FBX file to inspect')
    
    presets_parser = subparsers.add_parser('presets', help='List available presets')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--log-file', help='Log file path')
    
    return parser


def cmd_transfer(args) -> int:
    """Execute transfer command."""
    logger = get_logger('uv_transfer')
    
    config_manager = ConfigManager()
    
    if args.preset:
        preset_config = config_manager.get_preset(args.preset)
        if preset_config is None:
            logger.error(f"Preset not found: {args.preset}")
            return 1
        config = preset_config
    else:
        mode_map = {
            'direct': TransferMode.DIRECT,
            'spatial': TransferMode.SPATIAL,
            'interpolated': TransferMode.INTERPOLATED
        }
        
        algorithm_map = {
            'triangle_center': TransferAlgorithm.TRIANGLE_CENTER,
            'area_weighted': TransferAlgorithm.AREA_WEIGHTED,
            'normal_aware': TransferAlgorithm.NORMAL_AWARE,
        }
        
        config = TransferConfig(
            source_uv_channel=args.source_uv,
            target_uv_channel=args.target_uv,
            mode=mode_map[args.mode],
            algorithm=algorithm_map[args.algorithm],
            validate_source=not args.no_validate,
            validate_result=not args.no_validate,
            smooth_iterations=args.smooth,
            vertex_match_threshold=args.threshold
        )
    
    engine = UVTransferEngine(config=config)
    
    try:
        logger.info(f"Loading source: {args.source}")
        engine.load_source(args.source)
        
        logger.info(f"Loading target: {args.target}")
        engine.load_target(args.target)
        
        result = engine.transfer(
            source_mesh_name=args.source_mesh,
            target_mesh_name=args.target_mesh
        )
        
        if result.success:
            logger.info(f"Saving output: {args.output}")
            engine.save_result(args.output)
            
            print(f"\nTransfer completed successfully!")
            print(f"  Source vertices: {result.source_vertices}")
            print(f"  Target vertices: {result.target_vertices}")
            print(f"  Matched vertices: {result.matched_vertices} ({result.match_rate:.1%})")
            print(f"  Accuracy: {result.accuracy:.6f}")
            
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            
            return 0
        else:
            print("Transfer failed!")
            for error in result.errors:
                print(f"  Error: {error}")
            return 1
            
    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        return 1


def cmd_batch(args) -> int:
    """Execute batch command."""
    logger = get_logger('uv_transfer')
    
    processor = BatchProcessor()
    
    try:
        items = processor.load_batch_config(args.config)
        
        result = processor.process_batch(items, stop_on_error=args.stop_on_error)
        
        report = processor.generate_report(result, args.output)
        print(report)
        
        return 0 if result.failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1


def cmd_lod_chain(args) -> int:
    """Execute LOD chain command."""
    logger = get_logger('uv_transfer')
    
    processor = BatchProcessor()
    
    try:
        lod_files = processor.auto_detect_lod_files(args.directory)
        
        if len(lod_files) < 2:
            logger.error("Need at least 2 LOD files")
            return 1
        
        result = processor.process_lod_chain(
            lod_files,
            args.output,
            args.source_uv,
            args.target_uv
        )
        
        report = processor.generate_report(result)
        print(report)
        
        return 0 if result.failed == 0 else 1
        
    except Exception as e:
        logger.error(f"LOD chain processing failed: {e}")
        return 1


def cmd_view(args) -> int:
    """Execute view command."""
    logger = get_logger('uv_transfer')
    
    handler = FBXHandler()
    viewer = UVViewer()
    
    try:
        scene = handler.load(args.file)
        meshes = scene.get_all_meshes()
        
        if not meshes:
            logger.error("No meshes found in file")
            return 1
        
        mesh = meshes[0]
        uv_channel = mesh.get_uv_channel(args.uv_channel)
        
        if uv_channel is None:
            logger.error(f"UV channel {args.uv_channel} not found")
            return 1
        
        output_path = args.output or f"{mesh.name}_uv{args.uv_channel}.png"
        
        if args.show_islands:
            viewer.visualize_uv_islands(uv_channel, mesh.faces, output_path)
        else:
            viewer.visualize_uv(uv_channel, mesh.faces, output_path)
        
        print(f"UV visualization saved to: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"View failed: {e}")
        return 1


def cmd_validate(args) -> int:
    """Execute validate command."""
    logger = get_logger('uv_transfer')
    
    from .core.validator import UVValidator
    
    handler = FBXHandler()
    validator = UVValidator()
    
    try:
        scene = handler.load(args.file)
        meshes = scene.get_all_meshes()
        
        if not meshes:
            logger.error("No meshes found in file")
            return 1
        
        mesh = meshes[0]
        report = validator.generate_validation_report(mesh, args.uv_channel)
        
        print(report)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def cmd_compare(args) -> int:
    """Execute compare command."""
    logger = get_logger('uv_transfer')
    
    handler = FBXHandler()
    comparison = UVComparison()
    
    try:
        source_scene = handler.load(args.source)
        target_scene = handler.load(args.target)
        
        source_mesh = source_scene.get_all_meshes()[0]
        target_mesh = target_scene.get_all_meshes()[0]
        
        report = comparison.generate_comparison_report(
            source_mesh, target_mesh,
            args.source_uv, args.target_uv,
            output_dir=args.output_dir
        )
        
        print(f"\nComparison Results:")
        print(f"  Source: {report['source_mesh']} ({report['source_uv_channel']})")
        print(f"  Target: {report['target_mesh']} ({report['target_uv_channel']})")
        
        comp = report.get('comparison', {})
        print(f"\n  Source UV count: {comp.get('source_count', 'N/A')}")
        print(f"  Target UV count: {comp.get('target_count', 'N/A')}")
        
        if 'mean_distance' in comp:
            print(f"  Mean distance: {comp['mean_distance']:.6f}")
            print(f"  Max distance: {comp['max_distance']:.6f}")
        
        if 'comparison_image' in report:
            print(f"\n  Comparison image: {report['comparison_image']}")
        if 'heatmap_image' in report:
            print(f"  Heatmap image: {report['heatmap_image']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        return 1


def cmd_info(args) -> int:
    """Execute info command."""
    logger = get_logger('uv_transfer')
    
    handler = FBXHandler()
    
    try:
        scene = handler.load(args.file)
        
        print(f"\nFBX File: {args.file}")
        print(f"Meshes: {len(scene.meshes)}")
        print("-" * 50)
        
        for name, mesh in scene.meshes.items():
            info = handler.get_mesh_info(scene, name)
            print(f"\nMesh: {info['name']}")
            print(f"  Vertices: {info['vertex_count']}")
            print(f"  Faces: {info['face_count']}")
            print(f"  UV Channels: {len(info['uv_channels'])}")
            
            for uv_info in info['uv_channels']:
                print(f"    - {uv_info['name']}: {uv_info['uv_count']} UVs")
            
            print(f"  Has normals: {info['has_normals']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Info failed: {e}")
        return 1


def cmd_presets(args) -> int:
    """List available presets."""
    config_manager = ConfigManager()
    
    print("\nAvailable Presets:")
    print("-" * 50)
    
    for name in config_manager.get_preset_names():
        info = config_manager.get_preset_info(name)
        print(f"\n  {name}:")
        print(f"    {info['description']}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    log_level = 20
    if hasattr(args, 'verbose') and args.verbose:
        log_level = 10
    
    log_file = getattr(args, 'log_file', None)
    setup_logger(log_dir=os.path.dirname(log_file) if log_file else None)
    
    commands = {
        'transfer': cmd_transfer,
        'batch': cmd_batch,
        'lod-chain': cmd_lod_chain,
        'view': cmd_view,
        'validate': cmd_validate,
        'compare': cmd_compare,
        'info': cmd_info,
        'presets': cmd_presets,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
