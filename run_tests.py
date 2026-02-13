"""
Test runner script for UV Transfer Tool.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_tests(
    verbose: bool = True,
    coverage: bool = False,
    specific_tests: str = None
):
    """Run all tests."""
    import pytest
    
    args = []
    
    if verbose:
        args.append('-v')
    
    if coverage:
        args.extend(['--cov=uv_transfer', '--cov-report=html'])
    
    if specific_tests:
        args.append(specific_tests)
    else:
        args.append(str(Path(__file__).parent / 'tests'))
    
    return pytest.main(args)


def run_quick_test():
    """Run a quick test to verify installation."""
    print("=" * 60)
    print("UV Transfer Tool - Quick Test")
    print("=" * 60)
    
    print("\n1. Testing imports...")
    try:
        from uv_transfer import UVTransferEngine, UVValidator, FBXHandler
        print("   [OK] Core imports successful")
    except ImportError as e:
        print(f"   [FAIL] Import error: {e}")
        return False
    
    print("\n2. Testing FBX handler...")
    try:
        handler = FBXHandler()
        print(f"   [OK] FBX handler initialized")
        print(f"   Active backend: {handler.get_active_backend().value}")
        print(f"   Available backends: {[b.value for b in handler.get_available_backends()]}")
    except Exception as e:
        print(f"   [FAIL] FBX handler error: {e}")
        return False
    
    print("\n3. Testing transfer engine...")
    try:
        engine = UVTransferEngine()
        print("   [OK] Transfer engine initialized")
    except Exception as e:
        print(f"   [FAIL] Transfer engine error: {e}")
        return False
    
    print("\n4. Testing validator...")
    try:
        validator = UVValidator()
        print("   [OK] Validator initialized")
    except Exception as e:
        print(f"   [FAIL] Validator error: {e}")
        return False
    
    print("\n5. Testing with sample FBX files...")
    sample_dir = Path(__file__).parent
    fbx_files = list(sample_dir.glob("*.fbx"))
    
    if fbx_files:
        try:
            test_file = fbx_files[0]
            print(f"   Testing: {test_file.name}")
            
            scene = handler.load(str(test_file))
            print(f"   [OK] Loaded {len(scene.meshes)} mesh(es)")
            
            for name, mesh in scene.meshes.items():
                print(f"      - {name}: {mesh.vertex_count} vertices, {mesh.face_count} faces")
                for uv_name, uv_channel in mesh.uv_channels.items():
                    print(f"        UV: {uv_name} ({len(uv_channel.uv_coordinates)} UVs)")
        except Exception as e:
            print(f"   [FAIL] FBX loading error: {e}")
            return False
    else:
        print("   [SKIP] No sample FBX files found")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run UV Transfer Tool tests")
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--test', type=str, help='Run specific test file')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        exit_code = run_tests(
            verbose=True,
            coverage=args.coverage,
            specific_tests=args.test
        )
        sys.exit(exit_code)
