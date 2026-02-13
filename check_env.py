"""
Environment detection script for UV Transfer Tool.
Checks Python version, available FBX backends, and dependencies.
"""

import sys
import os
import subprocess
import importlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DependencyStatus:
    """Status of a dependency."""
    name: str
    installed: bool
    version: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EnvironmentReport:
    """Complete environment report."""
    python_version: str
    python_path: str
    platform: str
    fbx_backends: Dict[str, DependencyStatus]
    dependencies: Dict[str, DependencyStatus]
    recommendations: List[str]
    is_ready: bool


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        return True, version_str
    return False, version_str


def check_package(package_name: str, import_name: Optional[str] = None) -> DependencyStatus:
    """Check if a package is installed and get its version."""
    import_name = import_name or package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', None)
        return DependencyStatus(
            name=package_name,
            installed=True,
            version=version
        )
    except ImportError as e:
        return DependencyStatus(
            name=package_name,
            installed=False,
            error=str(e)
        )


def check_fbx_backends() -> Dict[str, DependencyStatus]:
    """Check available FBX backends."""
    backends = {}
    
    backends['fbx_official'] = check_package('fbx', 'fbx')
    backends['fbx_sdk_python'] = check_package('fbx-sdk-python', 'fbx')
    backends['pyfbx'] = check_package('pyfbx', 'pyfbx')
    backends['pyassimp'] = check_package('pyassimp', 'pyassimp')
    
    return backends


def check_core_dependencies() -> Dict[str, DependencyStatus]:
    """Check core dependencies."""
    deps = {}
    
    deps['numpy'] = check_package('numpy')
    deps['scipy'] = check_package('scipy')
    deps['matplotlib'] = check_package('matplotlib')
    deps['Pillow'] = check_package('Pillow', 'PIL')
    deps['pytest'] = check_package('pytest')
    deps['tqdm'] = check_package('tqdm')
    
    return deps


def get_recommendations(fbx_backends: Dict, dependencies: Dict) -> List[str]:
    """Generate recommendations based on environment status."""
    recommendations = []
    
    available_fbx = [name for name, status in fbx_backends.items() if status.installed]
    
    if not available_fbx:
        recommendations.append("No FBX backend available. Install one of the following:")
        recommendations.append("  - pip install fbx (Official Autodesk FBX SDK)")
        recommendations.append("  - pip install pyfbx (Alternative FBX parser)")
        recommendations.append("  - pip install pyassimp (Assimp wrapper)")
        recommendations.append("  The native parser will be used as fallback.")
    
    missing_deps = [name for name, status in dependencies.items() if not status.installed]
    if missing_deps:
        recommendations.append(f"Missing dependencies: {', '.join(missing_deps)}")
        recommendations.append("  Run: pip install -r requirements.txt")
    
    return recommendations


def run_environment_check() -> EnvironmentReport:
    """Run complete environment check."""
    print("=" * 60)
    print("UV Transfer Tool - Environment Check")
    print("=" * 60)
    
    py_ok, py_version = check_python_version()
    print(f"\nPython Version: {py_version}")
    print(f"Python Path: {sys.executable}")
    print(f"Platform: {sys.platform}")
    
    if not py_ok:
        print("WARNING: Python 3.8+ is required!")
    
    print("\n" + "-" * 40)
    print("FBX Backends:")
    print("-" * 40)
    
    fbx_backends = check_fbx_backends()
    for name, status in fbx_backends.items():
        if status.installed:
            version_str = f" (v{status.version})" if status.version else ""
            print(f"  [OK] {name}{version_str}")
        else:
            print(f"  [--] {name} - Not available")
    
    print("\n" + "-" * 40)
    print("Core Dependencies:")
    print("-" * 40)
    
    dependencies = check_core_dependencies()
    for name, status in dependencies.items():
        if status.installed:
            version_str = f" (v{status.version})" if status.version else ""
            print(f"  [OK] {name}{version_str}")
        else:
            print(f"  [--] {name} - Not installed")
    
    recommendations = get_recommendations(fbx_backends, dependencies)
    
    print("\n" + "-" * 40)
    print("Recommendations:")
    print("-" * 40)
    for rec in recommendations:
        print(f"  {rec}")
    
    has_fbx = any(s.installed for s in fbx_backends.values())
    has_deps = all(s.installed for s in dependencies.values())
    is_ready = py_ok and has_deps
    
    print("\n" + "=" * 60)
    if is_ready:
        print("Environment Status: READY")
    else:
        print("Environment Status: NEEDS ATTENTION")
    print("=" * 60)
    
    return EnvironmentReport(
        python_version=py_version,
        python_path=sys.executable,
        platform=sys.platform,
        fbx_backends=fbx_backends,
        dependencies=dependencies,
        recommendations=recommendations,
        is_ready=is_ready
    )


def test_fbx_read(file_path: str) -> bool:
    """Test FBX file reading capability."""
    print(f"\nTesting FBX read: {file_path}")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from uv_transfer.fbx.fbx_handler import FBXHandler
        
        handler = FBXHandler()
        print(f"Active backend: {handler.get_active_backend().value}")
        
        if handler.get_active_backend().value == 'none':
            print("ERROR: No FBX backend available!")
            return False
        
        scene = handler.load(file_path)
        print(f"Loaded {len(scene.meshes)} mesh(es)")
        
        for name, mesh in scene.meshes.items():
            print(f"  - {name}: {mesh.vertex_count} vertices, {mesh.face_count} faces")
            print(f"    UV channels: {list(mesh.uv_channels.keys())}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UV Transfer Tool Environment Check")
    parser.add_argument('--test-fbx', type=str, help='Test FBX file reading')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if args.test_fbx:
        success = test_fbx_read(args.test_fbx)
        sys.exit(0 if success else 1)
    else:
        report = run_environment_check()
        
        if args.json:
            import json
            output = {
                'python_version': report.python_version,
                'python_path': report.python_path,
                'platform': report.platform,
                'is_ready': report.is_ready,
                'fbx_backends': {
                    k: {'installed': v.installed, 'version': v.version}
                    for k, v in report.fbx_backends.items()
                },
                'dependencies': {
                    k: {'installed': v.installed, 'version': v.version}
                    for k, v in report.dependencies.items()
                },
                'recommendations': report.recommendations
            }
            print(json.dumps(output, indent=2))
        
        sys.exit(0 if report.is_ready else 1)
