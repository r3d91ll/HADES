#!/usr/bin/env python3
"""
Model Engine Test Runner

Comprehensive test runner for model engine components with detailed reporting,
coverage analysis, and performance benchmarking.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage reporting
    python run_tests.py --benchmark        # Run performance benchmarks
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --component vllm   # Test specific component
"""

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import json


class ModelEngineTestRunner:
    """Test runner for model engine components."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent.parent
        self.results = {}
        
    def run_pytest(self, args: List[str]) -> Dict[str, Any]:
        """Run pytest with specified arguments."""
        cmd = ["python", "-m", "pytest"] + args
        
        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "command": cmd
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "duration": time.time() - start_time,
                "command": cmd
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": time.time() - start_time,
                "command": cmd
            }
    
    def run_unit_tests(self, verbose: bool = False, component: str = None) -> Dict[str, Any]:
        """Run unit tests for model engine components."""
        print("\n🧪 Running Unit Tests...")
        
        test_patterns = []
        if component:
            if component == "core":
                test_patterns.append(str(self.test_dir / "core" / "test_*.py"))
            elif component in ["vllm", "haystack"]:
                test_patterns.append(str(self.test_dir / "engines" / component / "test_*.py"))
            else:
                print(f"❌ Unknown component: {component}")
                return {"success": False, "error": f"Unknown component: {component}"}
        else:
            test_patterns.extend([
                str(self.test_dir / "core" / "test_*.py"),
                str(self.test_dir / "engines" / "vllm" / "test_*.py"),
                str(self.test_dir / "engines" / "haystack" / "test_*.py")
            ])
        
        args = ["-m", "unit"] + test_patterns
        if verbose:
            args.extend(["-v", "-s"])
        
        result = self.run_pytest(args)
        self.results["unit_tests"] = result
        
        if result["success"]:
            print("✅ Unit tests passed!")
        else:
            print("❌ Unit tests failed!")
            if verbose:
                print(f"STDOUT:\n{result['stdout']}")
                print(f"STDERR:\n{result['stderr']}")
        
        return result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests for model engine components."""
        print("\n🔗 Running Integration Tests...")
        
        args = [
            "-m", "integration",
            str(self.test_dir / "test_integration.py")
        ]
        if verbose:
            args.extend(["-v", "-s"])
        
        result = self.run_pytest(args)
        self.results["integration_tests"] = result
        
        if result["success"]:
            print("✅ Integration tests passed!")
        else:
            print("❌ Integration tests failed!")
            if verbose:
                print(f"STDOUT:\n{result['stdout']}")
                print(f"STDERR:\n{result['stderr']}")
        
        return result
    
    def run_coverage_analysis(self, verbose: bool = False) -> Dict[str, Any]:
        """Run tests with coverage analysis."""
        print("\n📊 Running Coverage Analysis...")
        
        # Install coverage if not available
        try:
            import coverage
        except ImportError:
            print("Installing coverage...")
            subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], check=True)
        
        # Run tests with coverage
        args = [
            "--cov=src.components.model_engine",
            "--cov-report=term-missing",
            "--cov-report=html:test/coverage_html/model_engine",
            "--cov-report=json:test/coverage.json",
            str(self.test_dir)
        ]
        if verbose:
            args.extend(["-v", "-s"])
        
        result = self.run_pytest(args)
        self.results["coverage"] = result
        
        # Parse coverage results
        coverage_file = self.project_root / "test" / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                    print(f"📈 Total Coverage: {total_coverage:.1f}%")
                    
                    # Print per-file coverage
                    if verbose:
                        print("\n📋 Per-file Coverage:")
                        for filename, file_data in coverage_data.get("files", {}).items():
                            if "model_engine" in filename:
                                coverage_pct = file_data.get("summary", {}).get("percent_covered", 0)
                                print(f"  {filename}: {coverage_pct:.1f}%")
                
            except Exception as e:
                print(f"⚠️  Could not parse coverage data: {e}")
        
        if result["success"]:
            print("✅ Coverage analysis completed!")
        else:
            print("❌ Coverage analysis failed!")
        
        return result
    
    def run_performance_benchmarks(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance benchmarks for model engine components."""
        print("\n⚡ Running Performance Benchmarks...")
        
        # Create a simple benchmark script
        benchmark_script = """
import time
import statistics
from src.components.model_engine.core.processor import CoreModelEngine
from src.components.model_engine.engines.vllm.processor import VLLMModelEngine
from src.components.model_engine.engines.haystack.processor import HaystackModelEngine
from src.types.components.contracts import ModelEngineInput

def benchmark_engine(engine_class, config, input_data, iterations=5):
    engine = engine_class(config=config)
    times = []
    
    for _ in range(iterations):
        start = time.time()
        try:
            result = engine.process(input_data)
            times.append(time.time() - start)
        except Exception as e:
            print(f"Benchmark failed: {e}")
            return None
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0
    }

# Benchmark configurations
configs = {
    "core": {"engine_type": "vllm", "model_name": "test-model"},
    "vllm": {"server_url": "http://localhost:8000", "device": "cpu"},
    "haystack": {"server_url": "http://localhost:8080"}
}

# Test input
test_input = ModelEngineInput(
    requests=[{"request_id": "bench", "input_text": "benchmark test"}],
    model_config={}, batch_config={}, metadata={}
)

# Run benchmarks
engines = [
    (CoreModelEngine, configs["core"]),
    (VLLMModelEngine, configs["vllm"]),
    (HaystackModelEngine, configs["haystack"])
]

for engine_class, config in engines:
    print(f"Benchmarking {engine_class.__name__}...")
    stats = benchmark_engine(engine_class, config, test_input)
    if stats:
        print(f"  Mean: {stats['mean']:.3f}s")
        print(f"  Median: {stats['median']:.3f}s")
        print(f"  Range: {stats['min']:.3f}s - {stats['max']:.3f}s")
"""
        
        # Write and run benchmark
        benchmark_file = self.test_dir / "benchmark_temp.py"
        try:
            with open(benchmark_file, 'w') as f:
                f.write(benchmark_script)
            
            result = subprocess.run(
                [sys.executable, str(benchmark_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            benchmark_result = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            if verbose and benchmark_result["stdout"]:
                print(benchmark_result["stdout"])
            
        except Exception as e:
            benchmark_result = {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
        finally:
            # Cleanup
            if benchmark_file.exists():
                benchmark_file.unlink()
        
        self.results["benchmarks"] = benchmark_result
        
        if benchmark_result["success"]:
            print("✅ Performance benchmarks completed!")
        else:
            print("❌ Performance benchmarks failed!")
        
        return benchmark_result
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        print("🔍 Checking Dependencies...")
        
        required_packages = [
            "pytest",
            "pydantic",
            "typing_extensions"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies available!")
        return True
    
    def generate_report(self, output_file: str = None) -> None:
        """Generate a comprehensive test report."""
        print("\n📊 Generating Test Report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results,
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "total_duration": 0
            }
        }
        
        # Calculate summary statistics
        for test_type, result in self.results.items():
            if isinstance(result, dict) and "success" in result:
                report["summary"]["total_tests"] += 1
                if result["success"]:
                    report["summary"]["passed_tests"] += 1
                else:
                    report["summary"]["failed_tests"] += 1
                
                if "duration" in result:
                    report["summary"]["total_duration"] += result["duration"]
        
        # Print summary
        summary = report["summary"]
        print(f"\n📋 Test Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Duration: {summary['total_duration']:.2f}s")
        
        # Save report if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Report saved to: {output_path}")
        
        return report


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Model Engine Test Runner")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--component", choices=["core", "vllm", "haystack"], help="Test specific component")
    parser.add_argument("--report", help="Save report to file")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    
    args = parser.parse_args()
    
    runner = ModelEngineTestRunner()
    
    print("🚀 Model Engine Test Runner")
    print("=" * 50)
    
    # Check dependencies
    if not runner.check_dependencies():
        if args.check_deps:
            return 0
        print("\n❌ Cannot proceed with missing dependencies!")
        return 1
    
    if args.check_deps:
        return 0
    
    # Determine which tests to run
    run_all = not any([args.unit, args.integration, args.coverage, args.benchmark])
    
    success = True
    
    if args.unit or run_all:
        result = runner.run_unit_tests(verbose=args.verbose, component=args.component)
        success = success and result["success"]
    
    if args.integration or run_all:
        result = runner.run_integration_tests(verbose=args.verbose)
        success = success and result["success"]
    
    if args.coverage:
        result = runner.run_coverage_analysis(verbose=args.verbose)
        success = success and result["success"]
    
    if args.benchmark:
        result = runner.run_performance_benchmarks(verbose=args.verbose)
        success = success and result["success"]
    
    # Generate report
    runner.generate_report(output_file=args.report)
    
    # Final status
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())