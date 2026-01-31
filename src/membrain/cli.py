"""
Membrain CLI

Command-line interface for interacting with the Membrain service.
"""

import argparse
import sys


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="membrain",
        description="Neuromorphic Memory Bridge for LLM Agents",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the gRPC server")
    server_parser.add_argument(
        "--port", 
        type=int, 
        default=50051,
        help="Port to listen on (default: 50051)"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument(
        "--host",
        default="localhost:50051",
        help="Server address (default: localhost:50051)"
    )
    
    # Version command
    subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        from membrain.server import serve
        serve()
        return 0
    
    elif args.command == "status":
        # TODO: Implement status check
        print(f"Checking server at {args.host}...")
        print("Status check not yet implemented.")
        return 0
    
    elif args.command == "version":
        from membrain import __version__
        print(f"membrain {__version__}")
        return 0
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
