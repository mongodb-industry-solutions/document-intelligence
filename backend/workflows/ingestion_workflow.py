"""
Document Ingestion Workflow - Compatibility layer
This file maintains backward compatibility while delegating to the refactored modules
"""

# Re-export everything from the new modules for backward compatibility
from workflows.ingestion_builder import (
    create_ingestion_workflow,
    compile_workflow,
    create_initial_state
)

from agents.supervisor import SupervisorAgent
from agents.scanner import DocumentScannerAgent
from agents.evaluator import RelevanceEvaluatorAgent
from agents.extractor import VisualExtractorAgent
from processors.document_processor import DocumentProcessor

__all__ = [
    'create_ingestion_workflow',
    'compile_workflow',
    'create_initial_state',
    'SupervisorAgent',
    'DocumentScannerAgent',
    'RelevanceEvaluatorAgent',
    'VisualExtractorAgent',
    'DocumentProcessor'
]