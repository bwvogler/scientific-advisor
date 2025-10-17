"""
Helper utilities for the Scientific Advisor Agent.
"""

import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())

def extract_customer_info(text: str) -> Optional[str]:
    """Extract customer name from text using simple patterns."""
    # Common patterns for customer identification
    patterns = [
        r'customer[:\s]+([A-Za-z0-9\s]+)',
        r'client[:\s]+([A-Za-z0-9\s]+)',
        r'company[:\s]+([A-Za-z0-9\s]+)',
        r'organization[:\s]+([A-Za-z0-9\s]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def extract_project_info(text: str) -> Optional[str]:
    """Extract project name from text using simple patterns."""
    patterns = [
        r'project[:\s]+([A-Za-z0-9\s\-_]+)',
        r'study[:\s]+([A-Za-z0-9\s\-_]+)',
        r'experiment[:\s]+([A-Za-z0-9\s\-_]+)',
        r'protocol[:\s]+([A-Za-z0-9\s\-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def extract_dates(text: str) -> List[datetime]:
    """Extract dates from text."""
    import dateutil.parser
    
    # Common date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'[A-Za-z]+ \d{1,2}, \d{4}',  # Month DD, YYYY
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                date = dateutil.parser.parse(match)
                dates.append(date)
            except:
                continue
    
    return dates

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can break at a reasonable point
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_customer_name(customer: str) -> bool:
    """Validate customer name format."""
    if not customer or len(customer.strip()) < 2:
        return False
    
    # Allow alphanumeric, spaces, hyphens, underscores
    return bool(re.match(r'^[A-Za-z0-9\s\-_]+$', customer))

def validate_project_name(project: str) -> bool:
    """Validate project name format."""
    if not project or len(project.strip()) < 2:
        return False
    
    # Allow alphanumeric, spaces, hyphens, underscores
    return bool(re.match(r'^[A-Za-z0-9\s\-_]+$', project))

def create_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename."""
    import os
    
    metadata = {
        "original_filename": filename,
        "file_extension": os.path.splitext(filename)[1].lower(),
        "file_basename": os.path.splitext(filename)[0]
    }
    
    # Try to extract customer/project from filename
    customer = extract_customer_info(filename)
    if customer:
        metadata["extracted_customer"] = customer
    
    project = extract_project_info(filename)
    if project:
        metadata["extracted_project"] = project
    
    return metadata

def merge_metadata(base_metadata: Dict[str, Any], 
                  new_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata dictionaries, with new values taking precedence."""
    merged = base_metadata.copy()
    merged.update(new_metadata)
    return merged

def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary values."""
    try:
        result = data
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default

def calculate_importance_score(content: str, 
                             metadata: Dict[str, Any]) -> float:
    """Calculate importance score for content based on various factors."""
    score = 0.5  # Base score
    
    # Length factor (longer content might be more important)
    length_score = min(len(content) / 1000, 1.0) * 0.2
    score += length_score
    
    # Keyword factors
    important_keywords = [
        'urgent', 'critical', 'important', 'priority', 'deadline',
        'conclusion', 'recommendation', 'action', 'required'
    ]
    
    content_lower = content.lower()
    keyword_count = sum(1 for keyword in important_keywords if keyword in content_lower)
    keyword_score = min(keyword_count * 0.1, 0.3)
    score += keyword_score
    
    # Metadata factors
    if metadata.get("is_manual_entry"):
        score += 0.2  # Manual entries are typically more important
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))
