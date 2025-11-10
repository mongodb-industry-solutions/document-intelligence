"""
Scheduler Service for Daily Report Generation
Schedules and manages daily report generation for all industry/use case combinations
"""

import os
import logging
import asyncio
import json
from pathlib import Path
# https://github.com/dbader/schedule
import schedule
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set
import threading

from db.mongodb_connector import MongoDBConnector
from services.report_generator import ReportGenerator
from vogayeai.context_embeddings import VoyageContext3Embeddings
from cloud.aws.bedrock.client import BedrockClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportScheduler:
    """
    Manages scheduled report generation for all industry/use case combinations.
    Runs reports daily with staggered timing per industry.
    """
    
    def __init__(
        self,
        mongodb_connector: MongoDBConnector = None,
        embeddings_client: VoyageContext3Embeddings = None,
        bedrock_client=None
    ):
        """
        Initialize Report Scheduler.
        
        Args:
            mongodb_connector: MongoDB connector
            embeddings_client: VoyageAI embeddings client
            bedrock_client: AWS Bedrock client
        """
        # Initialize components if not provided
        if not mongodb_connector:
            self.mongodb_connector = MongoDBConnector()
        else:
            self.mongodb_connector = mongodb_connector
            
        if not embeddings_client:
            self.embeddings_client = VoyageContext3Embeddings()
        else:
            self.embeddings_client = embeddings_client
            
        if not bedrock_client:
            self.bedrock_client = BedrockClient()._get_bedrock_client()
        else:
            self.bedrock_client = bedrock_client
        
        # Initialize report generator
        self.report_generator = ReportGenerator(
            mongodb_connector=self.mongodb_connector,
            embeddings_client=self.embeddings_client,
            bedrock_client=self.bedrock_client
        )
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        
        # Validate initial state file at startup
        self._validate_initial_state_file()
        
        # Industry schedule (UTC times) - Weekly on Mondays
        self.industry_schedule = {
            "fsi": "04:00",           # 4:00 AM UTC every Monday
        }
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Check NODE_ENV to determine if jobs will be scheduled
        node_env = os.getenv("NODE_ENV", "").lower()
        jobs_enabled = node_env == "prod"
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        if jobs_enabled:
            logger.info(f"✅ Report scheduler started - Jobs ENABLED (NODE_ENV={node_env})")
        else:
            logger.info(f"✅ Report scheduler started - Jobs DISABLED (NODE_ENV={node_env or 'not set'}, jobs only run in 'prod')")
    
    def stop(self):
        """Stop the scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("❌ Report scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        # Get NODE_ENV to determine if jobs should be scheduled
        node_env = os.getenv("NODE_ENV", "").lower()
        
        # Only schedule jobs in production environment
        if node_env != "prod":
            logger.info(f"Skipping job scheduling - NODE_ENV={node_env} (jobs only run in 'prod' environment)")
            logger.info("Scheduled jobs configured! (no jobs scheduled)")
            # Run empty scheduler loop to keep thread alive but inactive
            while self.is_running:
                time.sleep(60)  # Check every minute
            return
        
        logger.info(f"Scheduling jobs for production environment (NODE_ENV={node_env})")
        
        # Clear existing jobs
        schedule.clear()
        
        # Schedule report generation for each industry
        # Weekly on Mondays
        for industry, time_str in self.industry_schedule.items():
            # Schedule for every Monday at specified time
            schedule.every().monday.at(time_str).do(
                self._run_industry_job, industry
            ).tag(industry)
            logger.info(f"📅 Scheduled {industry} reports weekly on Mondays at {time_str} UTC")
        
        # Schedule daily log cleanup at 3:00 AM UTC
        schedule.every().day.at("03:00").do(self._cleanup_logs_job)
        logger.info(f"🧹 Scheduled daily log cleanup at 03:00 AM UTC")
        
        # Schedule daily document state reset at 3:10 AM UTC (after logs cleanup)
        schedule.every().day.at("03:10").do(self._cleanup_documents_job)
        logger.info(f"🔄 Scheduled daily document state reset at 03:10 AM UTC")
        
        # Schedule an immediate test run (for development)
        if os.getenv("SCHEDULER_TEST_MODE", "false").lower() == "true":
            logger.info("🧪 Test mode: Scheduling immediate report generation")
            schedule.every(5).minutes.do(self._run_all_job)
        
        # Run the scheduler
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _run_industry_job(self, industry: str):
        """Wrapper for industry report generation (non-async)."""
        try:
            asyncio.run(self._generate_industry_reports(industry))
        except Exception as e:
            logger.error(f"Error in industry job {industry}: {e}")
    
    def _run_all_job(self):
        """Wrapper for all reports generation (non-async)."""
        try:
            asyncio.run(self._generate_all_reports())
        except Exception as e:
            logger.error(f"Error in all reports job: {e}")
    
    def _cleanup_logs_job(self):
        """Wrapper for daily log cleanup (non-async)."""
        try:
            asyncio.run(self._cleanup_old_logs())
        except Exception as e:
            logger.error(f"Error in log cleanup job: {e}")
    
    def _cleanup_documents_job(self):
        """Wrapper for daily document state reset (non-async)."""
        try:
            asyncio.run(self._cleanup_extra_documents())
        except Exception as e:
            logger.error(f"Error in document cleanup job: {e}")
    
    async def _generate_industry_reports(self, industry: str):
        """
        Generate reports for all use cases in an industry.
        
        Args:
            industry: Industry code
        """
        logger.info(f"🏭 Starting report generation for industry: {industry}")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get all available use cases
            use_cases = await self._get_use_cases_for_industry(industry)
            
            if not use_cases:
                logger.warning(f"No use cases found for industry: {industry}")
                return
            
            # Generate report for each use case
            success_count = 0
            error_count = 0
            
            for use_case in use_cases:
                try:
                    # Check if there are chunks for this combination
                    has_data = await self._has_chunks_for_combination(industry, use_case)
                    if not has_data:
                        logger.info(f"⏭️ Skipping {industry}/{use_case} - no data available")
                        continue
                    
                    logger.info(f"📊 Generating report for {industry}/{use_case}")
                    
                    # Generate report
                    result = await self.report_generator.generate_report(
                        industry=industry,
                        use_case=use_case,
                        report_date=datetime.now(timezone.utc).replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                    )
                    
                    if result.get("status") == "generated":
                        success_count += 1
                        logger.info(f"✅ Successfully generated {industry}/{use_case} report")
                    else:
                        error_count += 1
                        logger.error(f"❌ Failed to generate {industry}/{use_case} report: {result.get('error')}")
                    
                    # Small delay between reports to avoid overloading
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error generating report for {industry}/{use_case}: {e}")
            
            # Log summary
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                f"🏁 Completed {industry} report generation in {duration:.1f}s - "
                f"Success: {success_count}, Errors: {error_count}"
            )
            
            # Store scheduler run metadata
            self._store_scheduler_run(industry, success_count, error_count, duration)
            
        except Exception as e:
            logger.error(f"Critical error in industry report generation for {industry}: {e}")
    
    async def _generate_all_reports(self):
        """Generate reports for all industries (used for testing)."""
        logger.info("🌐 Generating reports for all industries")
        
        for industry in self.industry_schedule.keys():
            await self._generate_industry_reports(industry)
    
    async def _get_use_cases_for_industry(self, industry: str) -> List[str]:
        """
        Get all use cases that have data for a specific industry.
        
        Args:
            industry: Industry code
            
        Returns:
            List of use case codes
        """
        try:
            # Get all topic mappings (use cases)
            topics = list(self.mongodb_connector.industry_mappings_collection.find({
                "type": "topic",
                "enabled": True
            }))
            
            # Extract use case codes
            use_cases = [topic["code"] for topic in topics]
            
            return use_cases
            
        except Exception as e:
            logger.error(f"Error getting use cases: {e}")
            return []
    
    async def _has_chunks_for_combination(
        self,
        industry: str,
        use_case: str
    ) -> bool:
        """
        Check if there are chunks for an industry/use case combination.
        
        Args:
            industry: Industry code
            use_case: Use case code
            
        Returns:
            True if chunks exist
        """
        try:
            # Check if any chunks match the pattern
            count = self.mongodb_connector.collection.count_documents({
                "$or": [
                    {"metadata.path": {"$regex": f".*/{industry}/{use_case}/.*"}},
                    {"metadata.path": {"$regex": f".*/{industry}/.*{use_case}.*"}}
                ]
            }, limit=1)
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking chunks: {e}")
            return False
    
    def _store_scheduler_run(
        self,
        industry: str,
        success_count: int,
        error_count: int,
        duration: float
    ):
        """
        Store scheduler run metadata for monitoring.
        
        Args:
            industry: Industry code
            success_count: Number of successful reports
            error_count: Number of failed reports
            duration: Total duration in seconds
        """
        try:
            # This could be stored in a separate collection for monitoring
            # For now, we'll just log it
            logger.info(
                f"📈 Scheduler metrics - Industry: {industry}, "
                f"Success: {success_count}, Errors: {error_count}, "
                f"Duration: {duration:.1f}s"
            )
        except Exception as e:
            logger.error(f"Error storing scheduler run: {e}")
    
    async def _cleanup_old_logs(self):
        """
        Clean up old logs and checkpointer data from MongoDB collections.
        Runs daily at 3:00 AM UTC to manage database size.
        Deletes ALL documents from logs, logs_qa, and checkpointer collections.
        """
        logger.info("🧹 Starting daily log and checkpointer cleanup...")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Clean up ingestion workflow logs
            logs_result = self.mongodb_connector.logs_collection.delete_many({})
            logger.info(f"  Deleted {logs_result.deleted_count} ingestion workflow logs")
            
            # Clean up QA thinking process logs
            logs_qa_result = self.mongodb_connector.logs_qa_collection.delete_many({})
            logger.info(f"  Deleted {logs_qa_result.deleted_count} Q&A thinking process logs")
            
            # Clean up LangGraph checkpointer collections (conversation memory)
            checkpoint_writes_result = self.mongodb_connector.checkpoint_writes_collection.delete_many({})
            logger.info(f"  Deleted {checkpoint_writes_result.deleted_count} checkpoint writes")
            
            checkpoints_result = self.mongodb_connector.checkpoints_collection.delete_many({})
            logger.info(f"  Deleted {checkpoints_result.deleted_count} checkpoints")
            
            total_deleted = (
                logs_result.deleted_count + 
                logs_qa_result.deleted_count + 
                checkpoint_writes_result.deleted_count + 
                checkpoints_result.deleted_count
            )
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(
                f"✅ Cleanup completed in {duration:.1f}s - "
                f"Total deleted: {total_deleted} documents"
            )
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_initial_state_path(self) -> Optional[Path]:
        """
        Resolve the path to the initial state file with multiple fallback strategies.
        Works in both local development and Docker container environments.
        
        Returns:
            Path to the initial state file, or None if not found
        """
        filename = "documents_initial_state_dict.json"
        
        # Strategy 1: Environment variable override (highest priority)
        env_path = os.getenv("INITIAL_STATE_FILE_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
            else:
                logger.warning(f"⚠️ INITIAL_STATE_FILE_PATH set but file not found: {env_path}")
        
        # Strategy 2: Relative to this file (works in both dev and Docker)
        # In Docker: /services/scheduler.py -> / -> /data/seed/...
        # In Dev: backend/services/scheduler.py -> backend/ -> backend/data/seed/...
        relative_path = Path(__file__).parent.parent / "data" / "seed" / filename
        if relative_path.exists():
            return relative_path
        
        # Strategy 3: Absolute path in Docker container
        docker_path = Path("/data/seed") / filename
        if docker_path.exists():
            return docker_path
        
        # Strategy 4: Current working directory
        cwd_path = Path.cwd() / "data" / "seed" / filename
        if cwd_path.exists():
            return cwd_path
        
        # Strategy 5: Backend subdirectory from current working directory
        backend_path = Path.cwd() / "backend" / "data" / "seed" / filename
        if backend_path.exists():
            return backend_path
        
        return None
    
    def _validate_initial_state_file(self):
        """
        Validate that the initial state file exists and is readable at startup.
        Logs the resolved path for debugging Docker container issues.
        """
        logger.info("🔍 Validating initial document state file...")
        
        try:
            path = self._get_initial_state_path()
            
            if path is None:
                logger.error("❌ Initial state file NOT FOUND - searched multiple locations:")
                logger.error(f"  - Relative: {Path(__file__).parent.parent / 'data' / 'seed' / 'documents_initial_state_dict.json'}")
                logger.error(f"  - Docker: /data/seed/documents_initial_state_dict.json")
                logger.error(f"  - CWD: {Path.cwd() / 'data' / 'seed' / 'documents_initial_state_dict.json'}")
                logger.error("  ⚠️ Document state reset will be SKIPPED at 3:10 AM UTC")
                return
            
            # Try to load and validate the file
            with open(path, 'r') as f:
                state_data = json.load(f)
            
            if "documents" not in state_data:
                logger.error(f"❌ Initial state file found but missing 'documents' key: {path}")
                return
            
            doc_count = len(state_data["documents"])
            
            # Log success with full path
            logger.info(f"✅ Initial state file FOUND and VALID")
            logger.info(f"   📁 Path: {path.absolute()}")
            logger.info(f"   📊 Contains: {doc_count} blessed documents")
            logger.info(f"   🔄 Daily reset will maintain these {doc_count} documents")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Initial state file is not valid JSON: {e}")
        except Exception as e:
            logger.error(f"❌ Error validating initial state file: {e}")
    
    def _load_initial_state(self) -> Optional[Set[str]]:
        """
        Load the initial document state from JSON file.
        
        Returns:
            Set of document_ids that should be preserved, or None if file cannot be loaded
        """
        try:
            # Get the file path using the robust resolution strategy
            initial_state_path = self._get_initial_state_path()
            
            if initial_state_path is None:
                logger.error("❌ Initial state file not found")
                return None
            
            # Load and parse JSON
            with open(initial_state_path, 'r') as f:
                state_data = json.load(f)
            
            # Extract document_ids from the documents array
            if "documents" not in state_data:
                logger.error("❌ Initial state file missing 'documents' key")
                return None
            
            document_ids = {doc["document_id"] for doc in state_data["documents"]}
            
            logger.info(f"📋 Loaded {len(document_ids)} blessed document IDs from initial state")
            return document_ids
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse initial state JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading initial state: {e}")
            return None
    
    async def _cleanup_extra_documents(self):
        """
        Clean up documents that are not in the initial state baseline.
        Resets the system to the predetermined set of 20 documents.
        
        This maintains a clean demo environment by removing any documents
        added during testing, workshops, or demos.
        
        Deletes from:
        - documents collection (metadata)
        - chunks collection (embeddings)
        - assessments collection (evaluation cache)
        
        Does NOT touch:
        - gradings collection (Q&A relevance scores)
        - Document status (only presence/absence matters)
        """
        logger.info("🔄 Starting daily document state reset...")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Load blessed document IDs
            blessed_ids = self._load_initial_state()
            
            if blessed_ids is None:
                logger.error("⚠️ Cannot load initial state - skipping document cleanup for safety")
                return
            
            # Find all document_ids currently in the database
            current_doc_ids = set(
                self.mongodb_connector.documents_collection.distinct("document_id")
            )
            
            # Calculate extras (documents NOT in initial state)
            extra_doc_ids = current_doc_ids - blessed_ids
            
            if not extra_doc_ids:
                logger.info("✅ No extra documents found - system is in initial state")
                return
            
            logger.info(f"🗑️ Found {len(extra_doc_ids)} extra documents to remove")
            
            # Delete from documents collection
            docs_result = self.mongodb_connector.documents_collection.delete_many({
                "document_id": {"$in": list(extra_doc_ids)}
            })
            logger.info(f"  Deleted {docs_result.deleted_count} document metadata records")
            
            # Delete from chunks collection
            chunks_result = self.mongodb_connector.collection.delete_many({
                "document_id": {"$in": list(extra_doc_ids)}
            })
            logger.info(f"  Deleted {chunks_result.deleted_count} chunks")
            
            # Delete from assessments collection
            assessments_result = self.mongodb_connector.assessments_collection.delete_many({
                "document_id": {"$in": list(extra_doc_ids)}
            })
            logger.info(f"  Deleted {assessments_result.deleted_count} assessments")
            
            total_deleted = (
                docs_result.deleted_count + 
                chunks_result.deleted_count + 
                assessments_result.deleted_count
            )
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            logger.info(
                f"✅ Document state reset completed in {duration:.1f}s - "
                f"Removed {len(extra_doc_ids)} documents ({total_deleted} total records)"
            )
            
        except Exception as e:
            logger.error(f"❌ Error during document state reset: {e}")
    
    async def generate_report_manually(
        self,
        industry: str,
        use_case: str
    ) -> Dict[str, Any]:
        """
        Manually trigger report generation (for API endpoint).
        
        Args:
            industry: Industry code
            use_case: Use case code
            
        Returns:
            Report generation result
        """
        logger.info(f"🔧 Manual report generation triggered for {industry}/{use_case}")
        
        return await self.report_generator.generate_report(
            industry=industry,
            use_case=use_case,
            report_date=datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.
        
        Returns:
            Status information
        """
        return {
            "is_running": self.is_running,
            "industry_schedule": self.industry_schedule,
            "next_runs": self._get_next_run_times(),
            "thread_alive": self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }
    
    def _get_next_run_times(self) -> Dict[str, str]:
        """
        Get next scheduled run times for each industry.
        
        Returns:
            Dict of industry to next run time
        """
        next_runs = {}
        now = datetime.now(timezone.utc)
        
        for industry, time_str in self.industry_schedule.items():
            # Parse scheduled time
            hour, minute = map(int, time_str.split(':'))
            
            # Calculate next run
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                # If already passed today, schedule for tomorrow
                next_run += timedelta(days=1)
            
            next_runs[industry] = next_run.isoformat()
        
        return next_runs


# Global scheduler instance
_scheduler_instance: Optional[ReportScheduler] = None


def get_scheduler() -> ReportScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler_instance
    if not _scheduler_instance:
        _scheduler_instance = ReportScheduler()
    return _scheduler_instance


def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


def stop_scheduler():
    """Stop the global scheduler."""
    scheduler = get_scheduler()
    scheduler.stop()
