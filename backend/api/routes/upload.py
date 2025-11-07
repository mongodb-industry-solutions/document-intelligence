"""
Document Upload API Routes
Handles file uploads for containerized environments with industry-based organization

DEMO NOTE: This implementation uses a simplified local file storage approach within 
Docker containers, suitable for demonstrations. In enterprise environments, document 
intelligence solutions would typically include:
- Direct integration with enterprise content management systems (ECM)
- Distributed file systems or object storage for scalability
- Document versioning and audit trails
- Virus scanning and malware detection
- Advanced metadata extraction and tagging
- Integration with data loss prevention (DLP) systems
- Compliance with regulatory requirements (GDPR, HIPAA, etc.)
- Multi-tenancy and isolated storage per customer
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import logging
import os
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])


class Industry(str, Enum):
    """Supported industries for document organization"""
    FSI = "fsi"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    HEALTHCARE = "healthcare"
    MEDIA = "media"
    INSURANCE = "insurance"


# Get configuration from environment
DOCUMENT_STORAGE_PATH = os.getenv("DOCUMENT_STORAGE_PATH", "/docs")
ALLOWED_INDUSTRIES = os.getenv("ALLOWED_INDUSTRIES", "fsi,manufacturing,retail,healthcare,media,insurance").split(",")
DEFAULT_INDUSTRY = os.getenv("DEFAULT_INDUSTRY", "fsi")
MAX_UPLOAD_SIZE_MB = float(os.getenv("MAX_UPLOAD_SIZE_MB", "1"))
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_FILE_EXTENSIONS", "pdf,docx,doc").split(",")


@router.post("/documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    industry: Optional[Industry] = Form(default=None),
    use_case: Optional[str] = Form(default=None),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Upload documents for processing with industry and use-case based organization.
    
    In containerized environments, files are stored in:
    - Without use_case: /docs/{industry}/filename
    - With use_case: /docs/{industry}/{use_case}/filename
    
    Example:
    - FSI credit rating: /docs/fsi/credit_rating/document.pdf
    - FSI general: /docs/fsi/document.pdf
    - Manufacturing quality: /docs/manufacturing/quality_control/report.docx
    
    Args:
        files: List of files to upload
        industry: Industry category (fsi, manufacturing, retail, etc.)
        use_case: Optional use case subfolder (credit_rating, risk_assessment, etc.)
        background_tasks: Optional background task manager
    
    Returns:
        Upload status with file paths ready for ingestion
    """
    # Use default industry if not specified
    if not industry:
        industry = DEFAULT_INDUSTRY
    else:
        # Ensure we use the string value of the enum and lowercase
        industry = industry.value if hasattr(industry, 'value') else str(industry)
        industry = industry.lower()
    
    # Validate industry
    # Ensure industry is lowercase
    industry = industry.lower()
    
    if industry not in ALLOWED_INDUSTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid industry. Allowed values: {', '.join(ALLOWED_INDUSTRIES)}"
        )
    
    # Sanitize use_case if provided (remove special characters, lowercase, replace spaces)
    if use_case:
        use_case = use_case.lower().strip()
        use_case = "".join(c if c.isalnum() or c in "_-" else "_" for c in use_case)
        # Remove multiple underscores
        use_case = "_".join(filter(None, use_case.split("_")))
    
    uploaded_files = []
    errors = []
    
    # Build storage path based on industry and use_case
    if use_case:
        storage_path = Path(DOCUMENT_STORAGE_PATH) / industry / use_case
        display_path = f"{DOCUMENT_STORAGE_PATH}/{industry}/{use_case}"
    else:
        storage_path = Path(DOCUMENT_STORAGE_PATH) / industry
        display_path = f"{DOCUMENT_STORAGE_PATH}/{industry}"
    
    # Ensure directory exists
    storage_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Enforce single-file upload for simplicity
        if len(files) > 1:
            return {
                "status": "error",
                "message": "Only one file can be uploaded at a time",
                "errors": [{"filename": f.filename, "error": "too_many_files"} for f in files]
            }

        for file in files:
            # Validate file extension
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                errors.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue
            
            # Check file size
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            
            if file_size_mb > MAX_UPLOAD_SIZE_MB:
                errors.append({
                    "filename": file.filename,
                    "error": f"File too large ({file_size_mb:.2f}MB). Max size: {MAX_UPLOAD_SIZE_MB}MB"
                })
                continue
            
            # Use original filename (no timestamp prefix)
            file_path = storage_path / file.filename
            
            # Check if file already exists
            if file_path.exists():
                # Existing on storage: mark as duplicate but still provide ingestion_path so UI can trigger ingestion
                logger.warning(
                    f"Skipped upload for existing file: {file.filename} at {file_path}"
                )
                errors.append({
                    "filename": file.filename,
                    "error": "duplicate",
                    "message": "File already exists and was not uploaded",
                    "existing_path": str(file_path)
                })
                uploaded_files.append({
                    "filename": file.filename,
                    "size_mb": round((file_path.stat().st_size) / (1024 * 1024), 2) if file_path.exists() else None,
                    "industry": industry,
                    "use_case": use_case,
                    "container_path": str(file_path),
                    "ingestion_path": f"@local@{str(file_path)}"
                })
                continue
            
            # Save file to storage directory
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            file_info = {
                "filename": file.filename,
                "size_mb": file_size_mb,
                "industry": industry,
                "use_case": use_case,
                "container_path": str(file_path),
                "ingestion_path": f"@local@{str(file_path)}"
            }
            
            uploaded_files.append(file_info)
            log_msg = f"Uploaded {file.filename} to {file_path} for industry: {industry}"
            if use_case:
                log_msg += f", use case: {use_case}"
            logger.info(log_msg)
        
        # Prepare response
        if len(uploaded_files) > 0:
            message = f"Uploaded {len(uploaded_files)} files to {industry}"
            if use_case:
                message += f"/{use_case}"
            message += " directory"
        else:
            message = "No files were uploaded"
        
        if errors:
            message += f". {len(errors)} files failed to upload"
        
        response = {
            "status": "success" if (uploaded_files and not errors) else ("partial_success" if uploaded_files and errors else ("warning" if errors and not uploaded_files else "success")),
            "message": message,
            "industry": industry,
            "use_case": use_case,
            "storage_path": display_path,
            "files": uploaded_files,
            "errors": errors,
            "ready_for_ingestion": len(uploaded_files) > 0,
            "uploaded_count": len(uploaded_files),
            "skipped_count": len([e for e in errors if e.get("error") == "duplicate"])
        }
        
        # If files were uploaded successfully, provide ingestion guidance
        if uploaded_files:
            # Generate timestamp for workflow ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # For batch processing of this specific use case
            response["ingestion_command"] = {
                "endpoint": "/api/ingestion/start",
                "payload": {
                    "source_paths": [f"@local@{display_path}"],
                    "workflow_id": f"{industry}_{use_case}_{timestamp}" if use_case else f"{industry}_batch_{timestamp}"
                }
            }
            
            # For individual file processing
            response["individual_ingestion_paths"] = [
                f["ingestion_path"] for f in uploaded_files
            ]
        
        return response
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/direct-process")
async def upload_and_process(
    files: List[UploadFile] = File(...),
    industry: Optional[Industry] = Form(default=None),
    use_case: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    """
    Upload documents and immediately start processing.
    Returns workflow ID for tracking.
    
    This is a convenience endpoint that combines upload + ingestion.
    """
    from api.routes.ingestion import router as ingestion_router
    import uuid
    
    # Ensure industry is a string value and lowercase
    if industry:
        if hasattr(industry, 'value'):
            industry = industry.value
        industry = str(industry).lower()
    
    # First upload the files
    upload_result = await upload_documents(files, industry, use_case)
    
    if upload_result["status"] != "success" or not upload_result["files"]:
        return {
            "status": "error",
            "message": "No files uploaded successfully",
            "errors": upload_result.get("errors", [])
        }
    
    # Start ingestion for the uploaded files
    source_paths = [f["ingestion_path"] for f in upload_result["files"]]
    
    # Generate workflow ID with use_case if provided
    if use_case:
        workflow_id = f"{industry}_{use_case}_{uuid.uuid4().hex[:8]}"
    else:
        workflow_id = f"{industry}_{uuid.uuid4().hex[:8]}"
    
    # Import and call ingestion
    from workflows.ingestion_workflow import run_ingestion_workflow
    import asyncio
    
    # Run workflow in background
    asyncio.create_task(
        run_ingestion_workflow(
            source_paths=source_paths,
            workflow_id=workflow_id
        )
    )
    
    message = f"Processing {len(source_paths)} documents for {industry}"
    if use_case:
        message += f" - {use_case}"
    
    return {
        "status": "success",
        "workflow_id": workflow_id,
        "message": message,
        "industry": industry,
        "use_case": use_case,
        "files_uploaded": len(upload_result["files"]),
        "source_paths": source_paths,
        "monitor_url": f"/api/ingestion/status/{workflow_id}"
    }


@router.get("/industries")
async def get_supported_industries() -> Dict[str, Any]:
    """
    Get list of supported industries, their storage locations, and use cases.
    """
    industries = {}
    
    for industry in ALLOWED_INDUSTRIES:
        industry_path = Path(DOCUMENT_STORAGE_PATH) / industry
        
        # Check if directory exists and count files
        file_count = 0
        use_cases = []
        
        if industry_path.exists():
            # Count files directly in industry folder
            for ext in ALLOWED_EXTENSIONS:
                file_count += len(list(industry_path.glob(f"*.{ext}")))
            
            # Find use case subdirectories
            for subdir in industry_path.iterdir():
                if subdir.is_dir():
                    use_case_files = 0
                    for ext in ALLOWED_EXTENSIONS:
                        use_case_files += len(list(subdir.glob(f"*.{ext}")))
                    
                    if use_case_files > 0:
                        use_cases.append({
                            "name": subdir.name,
                            "file_count": use_case_files,
                            "storage_path": str(subdir),
                            "ingestion_path": f"@local@{subdir}"
                        })
                    file_count += use_case_files
        
        industries[industry] = {
            "name": industry.replace("_", " ").title(),
            "code": industry,
            "storage_path": f"{DOCUMENT_STORAGE_PATH}/{industry}",
            "ingestion_path": f"@local@{DOCUMENT_STORAGE_PATH}/{industry}",
            "file_count": file_count,
            "use_cases": use_cases,
            "is_default": industry == DEFAULT_INDUSTRY
        }
    
    return {
        "industries": industries,
        "default": DEFAULT_INDUSTRY,
        "storage_root": DOCUMENT_STORAGE_PATH,
        "example_use_cases": {
            "fsi": ["credit_rating", "risk_assessment", "compliance", "fraud_detection"],
            "manufacturing": ["quality_control", "supply_chain", "maintenance", "safety"],
            "retail": ["inventory", "customer_analytics", "pricing", "promotions"],
            "healthcare": ["patient_records", "clinical_trials", "billing", "compliance"],
            "media": ["content_analysis", "audience_insights", "advertising", "licensing"],
            "insurance": ["claims", "underwriting", "risk_assessment", "policy_management"]
        }
    }


@router.get("/storage-info")
async def get_storage_info() -> Dict[str, Any]:
    """
    Get complete storage configuration and status.
    """
    # Check available space
    storage_path = Path(DOCUMENT_STORAGE_PATH)
    
    if storage_path.exists():
        stat = os.statvfs(storage_path)
        free_space_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        total_space_mb = (stat.f_blocks * stat.f_frsize) / (1024 * 1024)
    else:
        free_space_mb = 0
        total_space_mb = 0
    
    # Count files per industry
    industry_stats = {}
    for industry in ALLOWED_INDUSTRIES:
        industry_path = Path(DOCUMENT_STORAGE_PATH) / industry
        if industry_path.exists():
            files = []
            for ext in ALLOWED_EXTENSIONS:
                files.extend(industry_path.glob(f"*.{ext}"))
            industry_stats[industry] = {
                "file_count": len(files),
                "path_exists": True
            }
        else:
            industry_stats[industry] = {
                "file_count": 0,
                "path_exists": False
            }
    
    return {
        "storage_path": DOCUMENT_STORAGE_PATH,
        "industries": industry_stats,
        "allowed_extensions": ALLOWED_EXTENSIONS,
        "max_file_size_mb": MAX_UPLOAD_SIZE_MB,
        "space": {
            "free_mb": round(free_space_mb, 2),
            "total_mb": round(total_space_mb, 2),
            "used_percentage": round((1 - free_space_mb/total_space_mb) * 100, 2) if total_space_mb > 0 else 0
        },
        "upload_endpoints": {
            "upload": "/api/upload/documents",
            "upload_and_process": "/api/upload/documents/direct-process"
        }
    }


@router.delete("/documents/{industry}")
async def delete_all_documents_in_folder(
    industry: str,
    use_case: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete all documents from an industry folder or use case subfolder.
    The folder structure is preserved, only files are deleted.
    
    Examples:
    - DELETE /api/upload/documents/fsi - deletes all files in /docs/fsi (including subdirectories)
    - DELETE /api/upload/documents/fsi?use_case=credit_rating - deletes all files in /docs/fsi/credit_rating
    """
    # Ensure industry is lowercase
    industry = industry.lower()
    
    if industry not in ALLOWED_INDUSTRIES:
        raise HTTPException(status_code=400, detail="Invalid industry")
    
    # Build target path
    if use_case:
        target_path = Path(DOCUMENT_STORAGE_PATH) / industry / use_case
        path_description = f"{industry}/{use_case}"
    else:
        target_path = Path(DOCUMENT_STORAGE_PATH) / industry
        path_description = industry
    
    if not target_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Path not found: {target_path}"
        )
    
    deleted_files = []
    errors = []
    
    try:
        # Find all files in the directory (recursively)
        for ext in ALLOWED_EXTENSIONS:
            for file_path in target_path.rglob(f"*.{ext}"):
                if file_path.is_file():
                    try:
                        relative_path = file_path.relative_to(Path(DOCUMENT_STORAGE_PATH))
                        file_path.unlink()
                        deleted_files.append(str(relative_path))
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        errors.append({
                            "file": str(file_path),
                            "error": str(e)
                        })
                        logger.error(f"Failed to delete {file_path}: {e}")
        
        return {
            "status": "success",
            "message": f"Deleted {len(deleted_files)} files from {path_description}",
            "path": str(target_path),
            "deleted_files": deleted_files,
            "errors": errors,
            "files_deleted": len(deleted_files),
            "files_failed": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Error during bulk delete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{industry}/{filename}")
async def delete_single_document(
    industry: str, 
    filename: str,
    use_case: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a specific document from an industry folder or use case subfolder.
    
    Examples:
    - DELETE /api/upload/documents/fsi/document.pdf
    - DELETE /api/upload/documents/fsi/document.pdf?use_case=credit_rating
    """
    # Ensure industry is lowercase
    industry = industry.lower()
    
    if industry not in ALLOWED_INDUSTRIES:
        raise HTTPException(status_code=400, detail="Invalid industry")
    
    # Build file path based on whether use_case is provided
    if use_case:
        file_path = Path(DOCUMENT_STORAGE_PATH) / industry / use_case / filename
    else:
        file_path = Path(DOCUMENT_STORAGE_PATH) / industry / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
        file_path.unlink()
        
        message = f"Deleted {filename} from {industry}"
        if use_case:
            message += f"/{use_case}"
            
        return {
            "status": "success",
            "message": message,
            "path": str(file_path),
            "file_size_mb": round(file_size, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_all_documents(
    confirm: bool = False
) -> Dict[str, Any]:
    """
    Delete ALL documents from ALL industries.
    Requires confirmation parameter to be true for safety.
    
    Example:
    - DELETE /api/upload/documents?confirm=true
    
    This is a dangerous operation that will delete all uploaded files!
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Add ?confirm=true to delete all documents."
        )
    
    total_deleted = 0
    deleted_by_industry = {}
    errors = []
    
    try:
        storage_path = Path(DOCUMENT_STORAGE_PATH)
        
        for industry in ALLOWED_INDUSTRIES:
            industry_path = storage_path / industry
            industry_deleted = []
            
            if industry_path.exists():
                for ext in ALLOWED_EXTENSIONS:
                    for file_path in industry_path.rglob(f"*.{ext}"):
                        if file_path.is_file():
                            try:
                                relative_path = file_path.relative_to(storage_path)
                                file_path.unlink()
                                industry_deleted.append(str(relative_path))
                                total_deleted += 1
                            except Exception as e:
                                errors.append({
                                    "file": str(file_path),
                                    "error": str(e)
                                })
            
            if industry_deleted:
                deleted_by_industry[industry] = {
                    "count": len(industry_deleted),
                    "files": industry_deleted
                }
        
        return {
            "status": "success",
            "message": f"Deleted {total_deleted} files across all industries",
            "total_deleted": total_deleted,
            "by_industry": deleted_by_industry,
            "errors": errors,
            "warning": "All uploaded documents have been deleted"
        }
        
    except Exception as e:
        logger.error(f"Error during complete cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{industry}")
async def list_documents(
    industry: str,
    use_case: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all documents in an industry folder or use case subfolder.
    
    Examples:
    - GET /api/upload/documents/fsi - lists all files in FSI
    - GET /api/upload/documents/fsi?use_case=credit_rating - lists files in credit_rating folder
    """
    # Ensure industry is lowercase
    industry = industry.lower()
    
    if industry not in ALLOWED_INDUSTRIES:
        raise HTTPException(status_code=400, detail="Invalid industry")
    
    # Build target path
    if use_case:
        target_path = Path(DOCUMENT_STORAGE_PATH) / industry / use_case
        path_description = f"{industry}/{use_case}"
    else:
        target_path = Path(DOCUMENT_STORAGE_PATH) / industry
        path_description = industry
    
    if not target_path.exists():
        return {
            "status": "success",
            "path": str(target_path),
            "description": path_description,
            "files": [],
            "total_files": 0,
            "total_size_mb": 0
        }
    
    files = []
    total_size = 0
    
    # List files based on scope
    for ext in ALLOWED_EXTENSIONS:
        if use_case:
            # For use_case, only list files directly in that folder
            file_pattern = target_path.glob(f"*.{ext}")
        else:
            # For industry, list all files recursively
            file_pattern = target_path.rglob(f"*.{ext}")
        
        for file_path in file_pattern:
            if file_path.is_file():
                file_stat = file_path.stat()
                relative_path = file_path.relative_to(target_path)
                
                # Determine use_case from path if at industry level
                file_use_case = None
                if not use_case and len(relative_path.parts) > 1:
                    file_use_case = relative_path.parts[0]
                
                files.append({
                    "filename": file_path.name,
                    "relative_path": str(relative_path),
                    "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "use_case": file_use_case or use_case,
                    "ingestion_path": f"@local@{file_path}"
                })
                total_size += file_stat.st_size
    
    # Sort by modified date (newest first)
    files.sort(key=lambda x: x["modified"], reverse=True)
    
    return {
        "status": "success",
        "path": str(target_path),
        "description": path_description,
        "files": files,
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }


@router.get("/documents")
async def list_all_documents() -> Dict[str, Any]:
    """
    List all documents across all industries.
    Provides a complete overview of the document storage.
    
    Example:
    - GET /api/upload/documents
    """
    all_files = []
    total_size = 0
    files_by_industry = {}
    
    storage_path = Path(DOCUMENT_STORAGE_PATH)
    
    for industry in ALLOWED_INDUSTRIES:
        industry_path = storage_path / industry
        industry_files = []
        industry_size = 0
        
        if industry_path.exists():
            for ext in ALLOWED_EXTENSIONS:
                for file_path in industry_path.rglob(f"*.{ext}"):
                    if file_path.is_file():
                        file_stat = file_path.stat()
                        relative_path = file_path.relative_to(storage_path)
                        
                        # Extract use_case if in subfolder
                        parts = relative_path.parts
                        file_use_case = parts[1] if len(parts) > 2 else None
                        
                        file_info = {
                            "filename": file_path.name,
                            "industry": industry,
                            "use_case": file_use_case,
                            "relative_path": str(relative_path),
                            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                            "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            "ingestion_path": f"@local@{file_path}"
                        }
                        
                        all_files.append(file_info)
                        industry_files.append(file_info)
                        total_size += file_stat.st_size
                        industry_size += file_stat.st_size
            
            if industry_files:
                files_by_industry[industry] = {
                    "count": len(industry_files),
                    "size_mb": round(industry_size / (1024 * 1024), 2),
                    "files": industry_files
                }
    
    # Sort all files by modified date (newest first)
    all_files.sort(key=lambda x: x["modified"], reverse=True)
    
    # Get summary statistics
    use_case_stats = {}
    for file in all_files:
        if file["use_case"]:
            key = f"{file['industry']}/{file['use_case']}"
            if key not in use_case_stats:
                use_case_stats[key] = {"count": 0, "size_mb": 0}
            use_case_stats[key]["count"] += 1
            use_case_stats[key]["size_mb"] += file["size_mb"]
    
    return {
        "status": "success",
        "total_files": len(all_files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "by_industry": files_by_industry,
        "by_use_case": use_case_stats,
        "recent_files": all_files[:10],  # Show 10 most recent files
        "storage_path": DOCUMENT_STORAGE_PATH
    }