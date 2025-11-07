"""
Claude Vision Document Extraction
Extracts document content as markdown with embedded visual descriptions
Pure vision understanding with detailed image descriptions
"""

from cloud.aws.bedrock.client import BedrockClient
from botocore.exceptions import ClientError
import json
import base64
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import io
from PIL import Image


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class VisualExtractionResult(BaseModel):
    """Extraction result with embedded visual descriptions"""
    markdown_content: str
    page_count: int
    has_visual_elements: bool = False
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)





class ClaudeVisionExtractor(BedrockClient):
    """
    Claude Vision Extractor with Visual Reference Tracking.
    Extracts content AND location data for visual grounding.
    """
    
    log: logging.Logger = logging.getLogger("ClaudeVisionExtractor")
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        model_id: str = os.getenv("BEDROCK_MODEL_ID")
    ):
        """Initialize extractor with visual tracking capabilities"""
        super().__init__(
            region_name=region_name
        )
        
        self.model_id = model_id
        self.bedrock_client = self._get_bedrock_client()
        
        logger.info(f"🎯 Claude Vision Extractor initialized with visual tracking")
    
    def extract_for_smart_ingestion(
        self,
        image_bytes: bytes,
        quick_scan: bool = True,
        industry: Optional[str] = None,
        topic: Optional[str] = None,
        industry_info: Optional[Dict[str, Any]] = None,
        topic_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Quick extraction for document relevance assessment during ingestion.
        
        Args:
            image_bytes: Single page image as bytes
            quick_scan: Whether to do a quick relevance scan
            
        Returns:
            Dictionary with relevance information
        """
        try:
            # Convert bytes to base64
            import base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Build context-aware assessment prompt
            context_section = ""
            if industry_info:
                context_section += f"\nIMPORTANT CONTEXT:\n"
                context_section += f"- Industry: {industry_info.get('full_name', industry)} ({industry})\n"
                if topic_info:
                    context_section += f"- Topic/Use Case: {topic_info.get('readable', topic)}\n"
                
                context_section += f"\nEvaluate if this document is relevant to EITHER:\n"
                context_section += f"1. The {industry_info.get('full_name', industry)} industry\n"
                if topic_info:
                    context_section += f"2. The topic of {topic_info.get('readable', topic)}\n"
                
                # Add examples
                if industry_info.get('examples'):
                    context_section += f"\nExamples of relevant {industry_info.get('full_name', industry)} content:\n"
                    context_section += "- " + "\n- ".join(industry_info['examples'][:5]) + "\n"
                
                if topic_info and topic_info.get('examples'):
                    context_section += f"\nExamples of relevant {topic_info.get('readable', topic)} content:\n"
                    context_section += "- " + "\n- ".join(topic_info['examples'][:3]) + "\n"
            
            # Quick relevance assessment prompt with strict criteria
            if context_section:
                # Context-aware assessment with explicit rejection criteria
                prompt = f"""Analyze this document image and assess its relevance STRICTLY.

{context_section}

CRITICAL ASSESSMENT RULES:
1. The document MUST be directly related to EITHER:
   - {industry_info.get('full_name', industry)} industry operations, business, or services
   - {topic_info.get('readable', topic)} as a business/professional topic

2. AUTOMATICALLY REJECT (score 0-49) if the document is:
   - Food/restaurant receipts or menus (unless for business expense reports)
   - Personal documents unrelated to business
   - Entertainment/leisure content
   - General consumer products unrelated to the industry/topic
   - Test or sample documents

3. ONLY ACCEPT (score 70+) if the document contains:
   - Business reports, analysis, or data related to {industry_info.get('full_name', industry)}
   - Professional documents about {topic_info.get('readable', topic)}
   - Official forms, statements, or records for the specified industry/topic

Scoring Guidelines:
- 0-49: NOT relevant - document has no genuine connection to {industry_info.get('full_name', industry)} or {topic_info.get('readable', topic)}
- 50-69: Marginally related - weak or indirect connection
- 70-89: Relevant - clear connection to either industry or topic
- 90-100: Highly relevant - strong connection to both industry and topic

BE STRICT: A pizza recipe, food receipt, or entertainment content should NEVER score above 49, regardless of creative reasoning.

Respond in JSON format:
{{
    "main_entity": "string",
    "document_category": "string",
    "key_topics": ["topic1", "topic2", "topic3"],
    "relevance_score": number,
    "reasoning": "explain ACTUAL relevance or why it's rejected",
    "matches_industry": boolean,
    "matches_topic": boolean
}}"""
            else:
                # General assessment without context
                prompt = """Analyze this document image and provide a quick assessment.

1. Main Entity: What organization/entity is this document about?
2. Document Category: What category does this document belong to? (report/form/letter/invoice/etc)
3. Key Topics: List 3-5 main topics/sections visible
4. Relevance Score: Rate the document quality and completeness (0-100)

Respond in JSON format:
{
    "main_entity": "string", 
    "document_category": "string",
    "key_topics": ["topic1", "topic2", "topic3"],
    "relevance_score": number,
    "reasoning": "brief explanation"
}"""

            # Prepare message for Claude
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }

            # Call Claude
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS", "8192")),
                    "temperature": 0.1,
                    "messages": [message]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # Try to parse JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"📊 Quick scan: {result.get('document_category', 'general')} - {result.get('relevance_score', 0)}% relevant")
                return result
            else:
                # Fallback if JSON parsing fails
                return {
                    "main_entity": "unknown",
                    "document_category": "unknown",
                    "key_topics": [],
                    "relevance_score": 50,
                    "reasoning": "Could not parse response"
                }
                
        except Exception as e:
            logger.error(f"Error in smart ingestion extraction: {e}")
            return {
                "main_entity": "unknown", 
                "document_category": "unknown",
                "key_topics": [],
                "relevance_score": 0,
                "reasoning": f"Extraction failed: {str(e)}"
            }
    
    def extract_with_visual_references(
        self,
        images: List[bytes],
        document_name: str = "document",
        document_id: str = None,
        extraction_hints: Optional[Dict[str, Any]] = None
    ) -> VisualExtractionResult:
        """
        Extract document content with visual reference tracking.
        Returns both markdown and exact locations of extracted content.
        
        Args:
            images: List of document page images
            document_name: Name of the document
            document_id: Unique document identifier
            extraction_hints: Optional hints for extraction
            
        Returns:
            VisualExtractionResult with markdown and visual references
        """
        logger.info(f"📸 Processing {len(images)} pages with visual reference tracking")
        
        if not document_id:
            import hashlib
            document_id = hashlib.sha256(document_name.encode()).hexdigest()[:16]
        
        all_markdown = []
        has_any_visual_elements = False
        extraction_stats = {
            "total_pages": len(images),
            "total_content_length": 0,
            "incomplete_pages": [],
            "retry_counts": [],
            "pages_with_visual_elements": []
        }
        
        for page_num, image_bytes in enumerate(images, 1):
            logger.info(f"📄 Extracting page {page_num}")
            
            # Extract page content as markdown
            # Reduce total attempts to 2 (initial + 1 retry)
            page_result = self._extract_page_as_markdown(
                image_bytes=image_bytes,
                page_num=page_num,
                total_pages=len(images),
                extraction_hints=extraction_hints,
                max_retries=1
            )
            
            # Process extraction result
            if page_result["success"]:
                all_markdown.append(page_result["markdown"])
                
                # Track if any page has visual elements
                if page_result.get("has_visual_elements", False):
                    has_any_visual_elements = True
                    extraction_stats["pages_with_visual_elements"].append(page_num)
                
                # Track extraction statistics
                if "extraction_stats" in page_result:
                    stats = page_result["extraction_stats"]
                    extraction_stats["total_content_length"] += stats.get("content_length", 0)
                    extraction_stats["retry_counts"].append(stats.get("retry_count", 0))
                    
                    if not page_result.get("is_complete", True):
                        extraction_stats["incomplete_pages"].append({
                            "page": page_num,
                            "reason": stats.get("validation_reason", "Unknown")
                        })
        
        # Combine markdown
        complete_markdown = "\n\n---\n\n".join(all_markdown)
        

        
        # Detect visual elements using proper markdown image syntax
        # This is more reliable than looking for keywords
        has_visual_elements = has_any_visual_elements or self._detect_visual_elements(complete_markdown)
        
        # Log extraction statistics
        if extraction_stats["incomplete_pages"]:
            logger.warning(f"⚠️ Extraction completed with {len(extraction_stats['incomplete_pages'])} potentially incomplete pages")
            for incomplete in extraction_stats["incomplete_pages"]:
                logger.warning(f"  - Page {incomplete['page']}: {incomplete['reason']}")
        
        avg_content_per_page = extraction_stats["total_content_length"] / len(images) if images else 0
        logger.info(f"📊 Extraction statistics:")
        logger.info(f"  - Total content: {extraction_stats['total_content_length']} chars")
        logger.info(f"  - Average per page: {avg_content_per_page:.0f} chars")
        logger.info(f"  - Pages with visual elements: {len(extraction_stats['pages_with_visual_elements'])}")
        logger.info(f"  - Total retries: {sum(extraction_stats['retry_counts'])}")
        
        result = VisualExtractionResult(
            markdown_content=complete_markdown,
            page_count=len(images),
            has_visual_elements=has_visual_elements,
            extraction_metadata={
                "model": self.model_id,
                "extraction_method": "vision_with_markdown_descriptions",
                "extraction_stats": extraction_stats
            }
        )
        
        logger.info(f"✅ Extraction complete. Has visual elements: {has_visual_elements}")
        return result
    
    def _extract_page_as_markdown(
        self,
        image_bytes: bytes,
        page_num: int,
        total_pages: int,
        extraction_hints: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Extract a single page with markdown and embedded image descriptions.
        Includes retry logic for incomplete extractions.
        
        Returns markdown content with visual elements described inline.
        """
        prompt = self._build_extraction_prompt(extraction_hints)
        
        # Encode image
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nThis is page {page_num} of {total_pages}."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
        
        # Request structured output
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": int(os.getenv("BEDROCK_MAX_TOKENS", "16384")),
            "temperature": 0.1,
            "system": """You are a document extraction expert. Your response must be in valid JSON format.

Extract all content from the image and convert to clean markdown with these rules:

1. For text content: Convert to proper markdown format
2. For tables: Convert to markdown table format
3. For images/charts/figures/diagrams: Use markdown image syntax with EXTREMELY DETAILED descriptions
   Format: ![COMPREHENSIVE description of the visual element](figure-X-descriptive-name.png)
   
CRITICAL INSTRUCTIONS FOR VISUAL ELEMENTS:
- NEVER use generic descriptions like "Revenue chart" or "Figure 1"
- ALWAYS include ALL visible data points, values, labels, and trends
- DESCRIBE the visual story the element is telling
- INCLUDE specific numbers, percentages, dates visible in the visual
- MENTION colors, shapes, and visual indicators (arrows, highlights, etc.)
- EXPLAIN what insights or conclusions the visual element conveys

Examples of GOOD image descriptions:
- ![Bar chart showing quarterly revenue growth from Q1 2023 ($1.2M) to Q4 2023 ($2.5M), with dark blue bars increasing in height from left to right. Each quarter shows approximately 15% growth over the previous quarter. Y-axis shows revenue in millions from $0 to $3M in $0.5M increments. X-axis shows quarters Q1-Q4 2023. A green trend line overlays the bars showing the upward trajectory.](figure-1-revenue-growth.png)

- ![Line graph depicting leverage ratios and FFO/Net Debt percentages from 2023-2026E. Two lines are shown: a red line for Net Debt/EBITDA starting at 4.3x in 2023 and declining to 4.0x by 2026E, and a blue line for FFO/Net Debt starting at 19% in 2023 and increasing to 22% by 2026E. The lines intersect at approximately 2024E. Grid lines appear at 0.5x intervals for leverage and 2% intervals for FFO/Net Debt.](figure-2-leverage-metrics.png)

- ![Stacked bar chart illustrating debt maturity profile from 2024-2030. Each bar represents annual debt maturities in millions. 2024: $200M (light blue), 2025: $350M (medium blue), 2026: $500M (dark blue), 2027: $300M (light blue), 2028: $450M (medium blue), 2029: $250M (dark blue), 2030: $150M (light blue). Total debt of $2.2B shown. A yellow line shows cumulative percentage of debt maturing, reaching 100% by 2030.](figure-3-debt-maturity.png)

Your response must be a JSON object with:
{
  "markdown": "Complete markdown content with DETAILED image descriptions following the above guidelines",
  "has_visual_elements": true/false (set to true if ANY charts, graphs, diagrams, or images are present)
}

IMPORTANT: Visual elements are critical for understanding documents. Extract EVERY detail visible in charts, graphs, and diagrams."""
        }
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # Log the full response for debugging
            logger.debug(f"Claude response length: {len(content)} characters")
            if len(content) < 1000:
                logger.warning(f"Short response from Claude: {content}")
            
            # Try to parse structured output
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{[\s\S]*"markdown"[\s\S]*\}', content)
                if json_match:
                    structured_data = json.loads(json_match.group())
                    markdown = structured_data.get("markdown", "")
                    has_visual_elements = structured_data.get("has_visual_elements", False)
                    
                    logger.debug(f"Successfully parsed JSON. Markdown length: {len(markdown)}")
                    
                    return {
                        "success": True,
                        "markdown": markdown,
                        "has_visual_elements": has_visual_elements,
                        "page_num": page_num
                    }
            except json.JSONDecodeError as e:
                logger.debug(f"Could not parse structured JSON: {e}")
                logger.debug(f"Raw content length: {len(content)}")
                # Log first 500 chars for debugging
                logger.debug(f"Content start: {content[:500]}...")
            
            # Fallback: Extract content more intelligently
            clean_content = content
            
            # Try to extract just the markdown content if we see JSON structure
            if '"markdown"' in content and '"has_visual_elements"' in content:
                try:
                    # First, try to find the boundaries of the markdown content
                    markdown_start = content.find('"markdown"')
                    if markdown_start != -1:
                        # Find the start of the actual markdown content
                        # Look for : and then the opening quote
                        colon_pos = content.find(':', markdown_start)
                        if colon_pos != -1:
                            # Find the opening quote after the colon
                            quote_start = content.find('"', colon_pos)
                            if quote_start != -1:
                                # Now find the end - look for the closing pattern
                                # We need to find '",\n  "has_visual_elements"' or similar
                                visual_pattern = '"has_visual_elements"'
                                visual_pos = content.find(visual_pattern, quote_start)
                                if visual_pos != -1:
                                    # Work backwards from visual_pos to find the closing quote
                                    search_pos = visual_pos - 1
                                    while search_pos > quote_start:
                                        if content[search_pos] == '"' and content[search_pos-1] != '\\':
                                            # Found the closing quote
                                            clean_content = content[quote_start+1:search_pos]
                                            break
                                        search_pos -= 1
                                    
                                    # Handle escaped characters
                                    clean_content = clean_content.replace('\\"', '"')
                                    clean_content = clean_content.replace('\\n', '\n')
                                    clean_content = clean_content.replace('\\t', '\t')
                                    clean_content = clean_content.replace('\\\\', '\\')
                                    logger.debug(f"Extracted markdown from JSON structure. Length: {len(clean_content)}")
                                else:
                                    logger.warning("Could not find has_visual_elements pattern")
                except Exception as e:
                    logger.warning(f"Error extracting markdown from JSON: {e}")
                    # Fall through to else clause
            else:
                # If the content is wrapped in quotes, remove them
                if content.startswith('"') and content.endswith('"'):
                    clean_content = content[1:-1]
                
                # Handle escaped characters
                clean_content = clean_content.replace('\\"', '"')
                clean_content = clean_content.replace('\\n', '\n')
                clean_content = clean_content.replace('\\t', '\t')
                clean_content = clean_content.replace('\\\\', '\\')
            
            logger.debug(f"Fallback parsing complete. Clean content length: {len(clean_content)}")
            
            # Detect if there are visual elements based on markdown image syntax
            has_visual_elements = bool(re.search(r'!\[.*?\]\(.*?\)', clean_content))
            
            # Validate extraction completeness
            validation = self._validate_extraction_completeness(clean_content, page_num)
            
            # If extraction seems incomplete and we have retries left, try again
            if not validation["is_complete"] and retry_count < max_retries:
                logger.warning(f"⚠️ Page {page_num} extraction seems incomplete: {validation['reason']}. Retrying... (attempt {retry_count + 2}/{max_retries + 1})")
                return self._extract_page_as_markdown(
                    image_bytes=image_bytes,
                    page_num=page_num,
                    total_pages=total_pages,
                    extraction_hints=extraction_hints,
                    retry_count=retry_count + 1,
                    max_retries=max_retries
                )
            
            # Log warning if still incomplete after retries
            if not validation["is_complete"]:
                logger.warning(f"⚠️ Page {page_num} extraction may be incomplete after {retry_count + 1} attempts: {validation['reason']}")
                logger.warning(f"Content length: {len(clean_content)} characters. This may be due to the token limit.")
            
            return {
                "success": True,
                "markdown": clean_content,
                "has_visual_elements": has_visual_elements,
                "page_num": page_num,
                "is_complete": validation["is_complete"],
                "extraction_stats": {
                    "content_length": len(clean_content),
                    "retry_count": retry_count,
                    "validation_reason": validation["reason"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num}: {e}")
            return {
                "success": False,
                "markdown": "",
                "extractions": [],
                "error": str(e)
            }
    
    def _build_extraction_prompt(self, hints: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for markdown extraction with image descriptions"""
        base_prompt = """Extract ALL content from this document image into markdown format.

IMPORTANT: For visual elements, use markdown image syntax with detailed descriptions.

Guidelines:
1. All text content: Convert to clean, structured markdown
2. Tables: Convert to markdown table format
3. Images/Charts/Figures/Diagrams: Use ![detailed description](filename.ext)
   - Be extremely detailed in descriptions
   - Include colors, data points, trends, relationships
   - Describe what it shows and why it's relevant
4. Preserve document structure and hierarchy

Example image descriptions:
- ![Bar chart showing monthly sales from Jan-Dec 2023, with values ranging from $50K to $150K, showing peak in November at $145K and lowest in February at $52K](sales-chart.png)
- ![Company logo featuring a blue circle with white mountain peaks inside, text "Alpine Corp" below in sans-serif font](logo.png)
- ![Workflow diagram with 4 connected boxes: Input → Processing → Validation → Output, with feedback loop from Validation back to Processing](workflow.png)

Focus on creating accessible, self-explanatory markdown content."""
        

        
        return base_prompt
    
    def _detect_visual_elements(self, markdown_content: str) -> bool:
        """
        Detect if markdown content contains visual elements.
        Checks for:
        - Markdown images: ![description](filename)
        - Markdown tables: | header | header | with separator line
        """
        # Check for markdown image syntax
        has_images = bool(re.search(r'!\[.*?\]\(.*?\)', markdown_content))
        
        # Check for markdown tables (header row with separator)
        # Looking for pattern like: |...|...\n|---|---|
        has_tables = bool(re.search(r'\|.*\|.*\n\s*\|[\s\-:|]+\|', markdown_content))
        
        return has_images or has_tables
    
    def _validate_extraction_completeness(self, content: str, page_num: int) -> Dict[str, Any]:
        """
        Validate if extraction seems complete.
        
        This is a simple heuristic-based validation that checks for common
        truncation patterns without being too strict.
        
        Args:
            content: Extracted markdown content
            page_num: Page number being validated
            
        Returns:
            Dictionary with is_complete flag and reason
        """
        # Minimum expected content length per page (very tolerant)
        MIN_CONTENT_LENGTH = 200  # Very low threshold to avoid false positives
        
        # Check for obvious truncation patterns
        truncation_patterns = [
            r'\.\.\.$',  # Ends with ellipsis
            r'[^.!?]\s*$',  # Doesn't end with sentence punctuation
            r'\w+\n*$',  # Ends with a single word
            r'^\s*$',  # Empty or whitespace only
            r'-\s*$',  # Ends with a dash
            r',\s*$',  # Ends with a comma
        ]
        
        # Special case: very short content
        if len(content.strip()) < MIN_CONTENT_LENGTH:
            return {
                "is_complete": False,
                "reason": f"Content too short ({len(content.strip())} chars, expected at least {MIN_CONTENT_LENGTH})"
            }
        
        # Check for truncation patterns
        content_end = content.strip()[-100:] if len(content.strip()) > 100 else content.strip()
        for pattern in truncation_patterns:
            if re.search(pattern, content_end):
                # Special handling: Tables and lists might legitimately end without punctuation
                if re.search(r'\|[^|]*$', content_end) or re.search(r'^[-*]\s', content_end.split('\n')[-1]):
                    continue  # Tables and lists are OK
                    
                return {
                    "is_complete": False,
                    "reason": f"Content appears truncated (pattern: {pattern})"
                }
        
        # Check for incomplete markdown structures
        # Count opening and closing markers
        open_code_blocks = content.count('```')
        if open_code_blocks % 2 != 0:
            return {
                "is_complete": False,
                "reason": "Unclosed code block detected"
            }
        
        # If we made it here, content seems complete enough
        return {
            "is_complete": True,
            "reason": "Content appears complete"
        }
    
