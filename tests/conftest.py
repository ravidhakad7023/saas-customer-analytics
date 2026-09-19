"""
Shared pytest configuration.
Adds src/ to path so tests can import modules without installation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
