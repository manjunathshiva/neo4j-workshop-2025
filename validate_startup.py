#!/usr/bin/env python3
"""
Startup validation script for Knowledge Graph RAG System.
Run this script to validate all cloud database connections and configuration.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.startup_validation import validate_startup_sync

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Knowledge Graph RAG System startup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_startup.py                    # Full validation with summary
  python validate_startup.py --quick           # Quick validation only
  python validate_startup.py --json            # Output as JSON
  python validate_startup.py --quiet           # No summary output
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Perform quick validation without establishing persistent connections"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Don't print validation summary to console"
    )
    parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output validation results as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        # Run validation
        results = validate_startup_sync(
            quick_mode=args.quick,
            print_summary=not args.quiet and not args.json
        )
        
        # Output JSON if requested
        if args.json:
            import json
            print(json.dumps(results, indent=2, default=str))
        
        # Exit with appropriate status code
        status = results.get("overall_status", "unknown")
        if status in ["critical_errors", "configuration_errors"]:
            if not args.quiet and not args.json:
                print("❌ System validation failed - check configuration and fix errors before proceeding")
            sys.exit(1)
        elif status == "degraded":
            if not args.quiet and not args.json:
                print("⚠️  System validation completed with warnings - some functionality may be limited")
            sys.exit(2)
        else:
            if not args.quiet and not args.json:
                print("✅ System validation successful - ready to start")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        sys.exit(1)