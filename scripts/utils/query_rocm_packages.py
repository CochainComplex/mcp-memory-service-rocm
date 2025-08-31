#!/usr/bin/env python3
"""
ROCm Package Query Module

Simple, reusable interface for querying highest available non-nightly ROCm packages.
Can be used as a module or standalone CLI tool.

Example usage:
    # As module
    from query_rocm_packages import query_packages
    result = query_packages("cp310", "6.4.1", ["torch", "tensorflow_rocm"])
    
    # As CLI
    python query_rocm_packages.py --python cp310 --rocm 6.4.1 --packages torch,tensorflow_rocm
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def normalize_rocm_version(version: str) -> str:
    """
    Normalize ROCm version for URL construction.
    Reused from fetch_all_rocm_packages.py for consistency.
    """
    # Don't normalize major versions (X.0) or older versions before 6.1
    if version in ['4.0', '5.0', '6.0'] or version.startswith(('3.', '4.0', '4.1', '4.2', '4.5', '5.')):
        return version
    
    # Only normalize patch versions .0 for newer releases (6.1+)
    if version.endswith('.0') and version.startswith('6.') and float(version.split('.')[1]) >= 1:
        return version[:-2]
    
    return version


def parse_wheel_filename(filename: str) -> Dict[str, Any]:
    """
    Parse a wheel filename to extract package information.
    Simplified version from fetch_all_rocm_packages.py.
    """
    # Remove .whl extension
    name = filename.replace('.whl', '')
    
    # Split by hyphens, but be careful with complex version strings
    parts = name.split('-')
    
    if len(parts) < 2:
        return {
            'filename': filename,
            'package_name': filename,
            'version': 'unknown',
            'python_version': None,
            'architecture': None
        }
    
    package_name = parts[0]
    version_part = parts[1]
    remaining_parts = parts[2:]
    
    # Look for python version patterns (cp310, cp311, etc.)
    python_version = None
    architecture = None
    
    for i, part in enumerate(remaining_parts):
        if part.startswith('cp') and part[2:].isdigit():
            python_version = part
            if i + 2 < len(remaining_parts):
                architecture = '-'.join(remaining_parts[i + 2:])
            break
        elif part in ['py2', 'py3', 'py2.py3']:
            python_version = part
            if i + 1 < len(remaining_parts):
                architecture = '-'.join(remaining_parts[i + 1:])
            break
    
    return {
        'filename': filename,
        'package_name': package_name,
        'version': version_part,
        'python_version': python_version,
        'architecture': architecture
    }


def is_nightly_package(package_name: str, version: str) -> bool:
    """
    Check if a package is a nightly build.
    
    Args:
        package_name: Name of the package
        version: Version string
        
    Returns:
        True if it's a nightly package
    """
    nightly_indicators = [
        'nightly',
        'dev',
        '.dev',
        'alpha',
        'beta',
        'rc',
        'pre'
    ]
    
    # Check package name
    if 'nightly' in package_name.lower():
        return True
    
    # Check version string
    version_lower = version.lower()
    for indicator in nightly_indicators:
        if indicator in version_lower:
            return True
    
    return False


def validate_inputs(python_version: str, rocm_version: str, package_names: List[str]) -> None:
    """
    Validate input parameters.
    
    Args:
        python_version: Python version (e.g., 'cp310')
        rocm_version: ROCm version (e.g., '6.4.1')
        package_names: List of package names
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate Python version format
    if not re.match(r'^(cp\d{2,3}|py[23]|py2\.py3)$', python_version):
        raise ValueError(f"Invalid Python version format: {python_version}. Expected format like 'cp310', 'py3', etc.")
    
    # Validate ROCm version format
    if not re.match(r'^\d+\.\d+(\.\d+)?$', rocm_version):
        raise ValueError(f"Invalid ROCm version format: {rocm_version}. Expected format like '6.4.1' or '6.4'")
    
    # Validate package names
    if not package_names or not all(package_names):
        raise ValueError("Package names cannot be empty")
    
    for package in package_names:
        if not re.match(r'^[a-zA-Z0-9_-]+$', package):
            raise ValueError(f"Invalid package name: {package}")


def query_packages(python_version: str, rocm_version: str, package_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Query highest available non-nightly versions of ROCm packages.
    
    Args:
        python_version: Python version (e.g., 'cp310')
        rocm_version: ROCm version (e.g., '6.4.1')  
        package_names: List of package names to query
        
    Returns:
        Dictionary with package info:
        {
            'torch': {
                'version': '2.7.1+rocm6.4.1',
                'filename': 'torch-2.7.1+rocm6.4.1-cp310-cp310-linux_x86_64.whl',
                'url': 'https://...',
                'found': True
            },
            'missing_pkg': {
                'found': False,
                'message': 'Package not found for Python cp310 in ROCm 6.4.1'
            }
        }
        
    Raises:
        ValueError: If input parameters are invalid
        requests.RequestException: If network request fails
    """
    # Validate inputs
    validate_inputs(python_version, rocm_version, package_names)
    
    # Normalize ROCm version for URL
    url_version = normalize_rocm_version(rocm_version)
    base_url = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{url_version}/"
    
    # Fetch repository contents
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch from {base_url}: {e}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all wheel files
    whl_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.whl'):
            whl_links.append(href)
    
    # Group packages by name
    packages_by_name = {}
    for filename in whl_links:
        parsed = parse_wheel_filename(filename)
        
        # Skip if not matching Python version
        if parsed['python_version'] != python_version:
            continue
            
        # Skip nightly packages
        if is_nightly_package(parsed['package_name'], parsed['version']):
            continue
        
        package_name = parsed['package_name']
        if package_name not in packages_by_name:
            packages_by_name[package_name] = []
        
        packages_by_name[package_name].append({
            'filename': filename,
            'version': parsed['version'],
            'url': urljoin(base_url, filename),
            'architecture': parsed['architecture']
        })
    
    # Sort by version and get highest for each package
    for package_name in packages_by_name:
        packages_by_name[package_name].sort(key=lambda x: x['version'], reverse=True)
    
    # Build result for requested packages
    result = {}
    for package_name in package_names:
        if package_name in packages_by_name and packages_by_name[package_name]:
            highest = packages_by_name[package_name][0]
            result[package_name] = {
                'version': highest['version'],
                'filename': highest['filename'],
                'url': highest['url'],
                'architecture': highest['architecture'],
                'found': True
            }
        else:
            result[package_name] = {
                'found': False,
                'message': f"Package not found for Python {python_version} in ROCm {rocm_version}"
            }
    
    return result


def format_output(results: Dict[str, Dict[str, Any]], output_format: str = "human") -> str:
    """
    Format query results for output.
    
    Args:
        results: Results from query_packages()
        output_format: "human" or "json"
        
    Returns:
        Formatted string
    """
    if output_format == "json":
        return json.dumps(results, indent=2)
    
    # Human-readable format
    lines = []
    for package_name, info in results.items():
        if info['found']:
            lines.append(f"{package_name}: {info['version']}")
            lines.append(f"  Download: {info['url']}")
        else:
            lines.append(f"{package_name}: NOT FOUND")
            lines.append(f"  {info['message']}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Query highest available non-nightly ROCm packages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --python cp310 --rocm 6.4.1 --packages torch
  %(prog)s --python cp311 --rocm 6.4 --packages torch,tensorflow_rocm --format json
        '''
    )
    
    parser.add_argument(
        '--python',
        required=True,
        help='Python version (e.g., cp310, cp311, py3)'
    )
    
    parser.add_argument(
        '--rocm',
        required=True,
        help='ROCm version (e.g., 6.4.1, 6.4)'
    )
    
    parser.add_argument(
        '--packages',
        required=True,
        help='Comma-separated list of package names'
    )
    
    parser.add_argument(
        '--format',
        choices=['human', 'json'],
        default='human',
        help='Output format (default: human)'
    )
    
    args = parser.parse_args()
    
    # Parse package names
    package_names = [pkg.strip() for pkg in args.packages.split(',')]
    
    try:
        # Query packages
        results = query_packages(args.python, args.rocm, package_names)
        
        # Output results
        output = format_output(results, args.format)
        print(output)
        
        # Set exit code based on whether all packages were found
        missing_count = sum(1 for info in results.values() if not info['found'])
        if missing_count > 0:
            sys.exit(1)  # Some packages not found
            
    except (ValueError, requests.RequestException) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()