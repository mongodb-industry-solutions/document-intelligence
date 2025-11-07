"""
FastAPI endpoints for scheduled report management
"""

import os
import logging
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.dependencies import get_mongodb_connector, get_embeddings_client, get_bedrock_client
from services.scheduler import get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportListRequest(BaseModel):
    """Request model for listing reports"""
    industry: Optional[str] = Field(None, description="Filter by industry")
    use_case: Optional[str] = Field(None, description="Filter by use case")
    limit: int = Field(10, description="Maximum number of reports to return")


class ReportInfo(BaseModel):
    """Report information model"""
    report_id: str
    industry: str
    use_case: str
    report_date: datetime
    file_path: str
    file_size_kb: float
    total_pages: int
    status: str
    generated_at: datetime
    chunk_count: int
    document_count: int


class GenerateReportRequest(BaseModel):
    """Request model for manual report generation"""
    industry: str = Field(..., description="Industry code (e.g., 'fsi', 'healthcare')")
    use_case: str = Field(..., description="Use case code (e.g., 'credit_rating', 'risk_assessment')")


@router.get("/list", response_model=List[ReportInfo])
async def list_reports(
    industry: Optional[str] = None,
    use_case: Optional[str] = None,
    limit: int = 10,
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    List available scheduled reports with optional filtering.
    
    Query Parameters:
    - industry: Filter by industry code
    - use_case: Filter by use case code
    - limit: Maximum number of reports to return (default: 10)
    """
    try:
        # Build filter query
        filter_query = {"status": "generated"}
        if industry:
            filter_query["industry"] = industry
        if use_case:
            filter_query["use_case"] = use_case
        
        # Get reports from MongoDB
        reports = list(mongodb_connector.scheduled_reports_collection.find(
            filter_query,
            limit=limit
        ).sort("generated_at", -1))
        
        # Convert to response model
        report_list = []
        for report in reports:
            report_list.append(ReportInfo(
                report_id=str(report["_id"]),
                industry=report["industry"],
                use_case=report["use_case"],
                report_date=report["report_date"],
                file_path=report["file_path"],
                file_size=report["file_size"],
                status=report["status"],
                generated_at=report["generated_at"],
                chunk_count=report["chunk_count"],
                document_count=report["document_count"],
                report_metadata=report["report_metadata"]
            ))
        
        logger.info(f"Retrieved {len(report_list)} reports")
        return report_list
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Download a specific report PDF file.
    
    Path Parameters:
    - report_id: MongoDB ObjectId of the report, or "seed" for seed reports
    """
    try:
        # Handle seed report downloads
        if report_id == "seed":
            raise HTTPException(
                status_code=400, 
                detail="Seed report downloads should use the latest endpoint"
            )
        
        from bson import ObjectId
        
        # Get report metadata
        report = mongodb_connector.scheduled_reports_collection.find_one({
            "_id": ObjectId(report_id)
        })
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        file_path = report.get("file_path")
        if not file_path or not os.path.exists(file_path):
            # File doesn't exist (container restart scenario), try seed report fallback
            industry = report.get("industry")
            use_case = report.get("use_case")
            seed_report_path = f"/backend/data/seed/reports/{industry}/{use_case}/seed_{use_case}_report.pdf"
            
            if os.path.exists(seed_report_path):
                logger.info(f"Using seed report fallback for download: {seed_report_path}")
                file_path = seed_report_path
            else:
                raise HTTPException(status_code=404, detail="Report file not found and no seed report available")
        
        # Return file
        filename = os.path.basename(file_path)
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/seed/{industry}/{use_case}/download")
async def download_seed_report(
    industry: str,
    use_case: str
):
    """
    Download a seed report PDF file.
    
    Path Parameters:
    - industry: Industry code (e.g., 'fsi', 'healthcare')
    - use_case: Use case code (e.g., 'credit_rating', 'risk_assessment')
    """
    try:
        seed_report_path = f"/backend/data/seed/reports/{industry}/{use_case}/seed_{use_case}_report.pdf"
        
        if not os.path.exists(seed_report_path):
            raise HTTPException(status_code=404, detail="Seed report not found")
        
        # Return file
        filename = f"seed_{use_case}_report.pdf"
        return FileResponse(
            path=seed_report_path,
            filename=filename,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error downloading seed report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/seed/{industry}/{use_case}/preview")
async def preview_seed_report(
    industry: str,
    use_case: str
):
    """
    Preview a seed report PDF file in browser (inline display, not download).
    
    Path Parameters:
    - industry: Industry code (e.g., 'fsi', 'healthcare')
    - use_case: Use case code (e.g., 'credit_rating', 'risk_assessment')
    """
    try:
        seed_report_path = f"/backend/data/seed/reports/{industry}/{use_case}/seed_{use_case}_report.pdf"
        
        if not os.path.exists(seed_report_path):
            raise HTTPException(status_code=404, detail="Seed report not found")
        
        # Read PDF file
        with open(seed_report_path, "rb") as f:
            pdf_content = f.read()
        
        # Return PDF with inline disposition for browser preview
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline",
                "Content-Type": "application/pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"Error previewing seed report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/preview")
async def preview_report(
    report_id: str,
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Get report preview URL for iframe display.
    
    Path Parameters:
    - report_id: MongoDB ObjectId of the report
    """
    try:
        from bson import ObjectId
        
        # Get report metadata
        report = mongodb_connector.scheduled_reports_collection.find_one({
            "_id": ObjectId(report_id)
        })
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        file_path = report.get("file_path")
        if not file_path or not os.path.exists(file_path):
            # File doesn't exist (container restart scenario), try seed report fallback
            industry = report.get("industry")
            use_case = report.get("use_case")
            seed_report_path = f"/backend/data/seed/reports/{industry}/{use_case}/seed_{use_case}_report.pdf"
            
            if os.path.exists(seed_report_path):
                logger.info(f"Using seed report fallback for preview: {seed_report_path}")
                file_path = seed_report_path
            else:
                raise HTTPException(status_code=404, detail="Report file not found and no seed report available")
        
        # Read PDF file
        with open(file_path, "rb") as f:
            pdf_content = f.read()
        
        # Return PDF with inline disposition for preview
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline",
                "Content-Type": "application/pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"Error previewing report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_report_manually(
    request: GenerateReportRequest,
    mongodb_connector=Depends(get_mongodb_connector),
    embeddings_client=Depends(get_embeddings_client),
    bedrock_client=Depends(get_bedrock_client)
):
    """
    Manually trigger report generation for a specific industry/use case.
    
    This endpoint allows on-demand report generation outside of the regular schedule.
    """
    try:
        logger.info(f"Manual report generation requested for {request.industry}/{request.use_case}")
        
        # Get scheduler instance
        scheduler = get_scheduler()
        
        # Generate report
        result = await scheduler.generate_report_manually(
            industry=request.industry,
            use_case=request.use_case
        )
        
        if result.get("status") == "generated":
            return {
                "status": "success",
                "message": f"Report generated successfully for {request.industry}/{request.use_case}",
                "report_id": str(result.get("_id", "")),
                "file_path": result.get("file_path"),
                "file_size": result.get("file_size")
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to generate report: {result.get('error', 'Unknown error')}",
                "error": result.get("error")
            }
        
    except Exception as e:
        logger.error(f"Error generating report manually: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-adhoc")
async def generate_adhoc_report(
    request: GenerateReportRequest,
    mongodb_connector=Depends(get_mongodb_connector),
    embeddings_client=Depends(get_embeddings_client),
    bedrock_client=Depends(get_bedrock_client)
):
    """
    Generate an ad-hoc report and store it both as a scheduled report and as a seed report.
    
    This endpoint is used when no report is available and the user wants to generate one on-demand.
    The report will be stored in both the regular reports location and as a seed report for fallback.
    """
    try:
        logger.info(f"Ad-hoc report generation requested for {request.industry}/{request.use_case}")
        
        # Get scheduler instance
        scheduler = get_scheduler()
        
        # Generate report
        result = await scheduler.generate_report_manually(
            industry=request.industry,
            use_case=request.use_case
        )
        
        if result.get("status") == "generated":
            # Also copy to seed location for fallback
            source_path = result.get("file_path")
            seed_dir = f"/backend/data/seed/reports/{request.industry}/{request.use_case}"
            seed_path = f"{seed_dir}/seed_{request.use_case}_report.pdf"
            
            # Create seed directory if it doesn't exist
            os.makedirs(seed_dir, exist_ok=True)
            
            # Copy report to seed location
            import shutil
            shutil.copy2(source_path, seed_path)
            
            logger.info(f"Report copied to seed location: {seed_path}")
            
            return {
                "status": "success",
                "message": f"Ad-hoc report generated and stored for {request.industry}/{request.use_case}",
                "report_id": str(result.get("_id", "")),
                "file_path": result.get("file_path"),
                "seed_path": seed_path,
                "file_size_kb": result.get("file_size_kb")
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to generate ad-hoc report: {result.get('error', 'Unknown error')}",
                "error": result.get("error")
            }
        
    except Exception as e:
        logger.error(f"Error generating ad-hoc report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_scheduler_status():
    """
    Get the current status of the report scheduler.
    
    Returns information about:
    - Whether the scheduler is running
    - Industry schedule times
    - Next scheduled run times
    """
    try:
        scheduler = get_scheduler()
        status = scheduler.get_scheduler_status()
        
        return {
            "scheduler_status": status,
            "current_time_utc": datetime.now(timezone.utc).isoformat(),
            "message": "Scheduler is " + ("running" if status["is_running"] else "stopped")
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{industry}/{use_case}")
async def get_latest_report(
    industry: str,
    use_case: str,
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Get the latest report for a specific industry/use case combination.
    Falls back to seed report if no generated report is available.
    
    Path Parameters:
    - industry: Industry code (e.g., 'fsi', 'healthcare')
    - use_case: Use case code (e.g., 'credit_rating', 'risk_assessment')
    """
    try:
        # First, try to find the latest generated report (exclude orphaned)
        report = mongodb_connector.scheduled_reports_collection.find_one({
            "industry": industry,
            "use_case": use_case,
            "status": "generated"
        }, sort=[("generated_at", -1)])
        
        if report:
            # Check if the file actually exists on the filesystem
            file_path = report.get("file_path")
            if file_path and os.path.exists(file_path):
                # File exists, return the generated report
                return ReportInfo(
                    report_id=str(report["_id"]),
                    industry=report["industry"],
                    use_case=report["use_case"],
                    report_date=report["report_date"],
                    file_path=file_path,
                    file_size_kb=report["file_size_kb"],
                    total_pages=report["total_pages"],
                    status=report["status"],
                    generated_at=report["generated_at"],
                    chunk_count=report["chunk_count"],
                    document_count=report["document_count"]
                )
            else:
                # File doesn't exist (container restart scenario), fall through to seed report
                logger.warning(f"Report file not found on filesystem: {file_path}, falling back to seed report")
        
        # Fallback to seed report
        seed_report_path = f"/backend/data/seed/reports/{industry}/{use_case}/seed_{use_case}_report.pdf"
        if os.path.exists(seed_report_path):
            # Get file size
            file_size = os.path.getsize(seed_report_path)
            file_size_kb = round(file_size / 1024, 2)
            
            # Create a mock report info for seed report
            return ReportInfo(
                report_id="seed",
                industry=industry,
                use_case=use_case,
                report_date=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
                file_path=seed_report_path,
                file_size_kb=file_size_kb,
                total_pages=4,  # Default for seed reports
                status="seed",
                generated_at=datetime.now(timezone.utc),
                chunk_count=0,
                document_count=0
            )
        
        # No report available at all
        raise HTTPException(
            status_code=404, 
            detail=f"No report found for {industry}/{use_case}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/industries")
async def get_available_industries(
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Get list of industries and use cases that have available reports.
    """
    try:
        # Aggregate to get unique industry/use case combinations
        pipeline = [
            {"$match": {"status": "generated"}},
            {"$group": {
                "_id": {
                    "industry": "$industry",
                    "use_case": "$use_case"
                },
                "latest_report": {"$max": "$generated_at"},
                "report_count": {"$sum": 1}
            }},
            {"$sort": {"_id.industry": 1, "_id.use_case": 1}}
        ]
        
        results = list(mongodb_connector.scheduled_reports_collection.aggregate(pipeline))
        
        # Organize by industry
        industries = {}
        for result in results:
            industry = result["_id"]["industry"]
            use_case = result["_id"]["use_case"]
            
            if industry not in industries:
                industries[industry] = []
            
            industries[industry].append({
                "use_case": use_case,
                "latest_report": result["latest_report"],
                "report_count": result["report_count"]
            })
        
        return {
            "industries": industries,
            "total_combinations": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error getting available industries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-orphaned")
async def cleanup_orphaned_reports(
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Clean up orphaned report records where the file no longer exists on the filesystem.
    This is useful after container restarts where MongoDB records exist but files are missing.
    """
    try:
        # Find all reports
        reports = list(mongodb_connector.scheduled_reports_collection.find({
            "status": "generated"
        }))
        
        orphaned_count = 0
        for report in reports:
            file_path = report.get("file_path")
            if file_path and not os.path.exists(file_path):
                # Check if seed report exists as fallback
                industry = report.get("industry")
                use_case = report.get("use_case")
                seed_report_path = f"/backend/data/seed/reports/{industry}/{use_case}/seed_{use_case}_report.pdf"
                
                if not os.path.exists(seed_report_path):
                    # No fallback available, mark as orphaned
                    mongodb_connector.scheduled_reports_collection.update_one(
                        {"_id": report["_id"]},
                        {"$set": {"status": "orphaned"}}
                    )
                    orphaned_count += 1
                    logger.info(f"Marked orphaned report: {report['_id']}")
        
        return {
            "status": "success",
            "message": f"Cleaned up {orphaned_count} orphaned reports",
            "orphaned_count": orphaned_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up orphaned reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for the reports service"""
    return {
        "status": "healthy",
        "service": "scheduled-reports",
        "version": "1.0.0",
        "description": "Scheduled reports service for document intelligence"
    }
