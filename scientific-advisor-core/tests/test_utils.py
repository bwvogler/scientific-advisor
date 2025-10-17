"""
Tests for utility functions.
"""

import pytest
from datetime import datetime
from src.utils.helpers import (
    generate_id, extract_customer_info, extract_project_info,
    clean_text, truncate_text, format_file_size,
    validate_customer_name, validate_project_name,
    create_metadata_from_filename, calculate_importance_score
)

def test_generate_id():
    """Test ID generation."""
    id1 = generate_id()
    id2 = generate_id()
    
    assert len(id1) == 36  # UUID length
    assert id1 != id2
    assert isinstance(id1, str)

def test_extract_customer_info():
    """Test customer info extraction."""
    text1 = "Customer: Acme Corp"
    text2 = "Client information for BioTech Inc"
    text3 = "Company details: XYZ Labs"
    text4 = "No customer info here"
    
    assert extract_customer_info(text1) == "Acme Corp"
    assert extract_customer_info(text2) == "BioTech Inc"
    assert extract_customer_info(text3) == "XYZ Labs"
    assert extract_customer_info(text4) is None

def test_extract_project_info():
    """Test project info extraction."""
    text1 = "Project: Alpha Study"
    text2 = "Study information for Beta Protocol"
    text3 = "Experiment: Gamma Analysis"
    text4 = "No project info here"
    
    assert extract_project_info(text1) == "Alpha Study"
    assert extract_project_info(text2) == "Beta Protocol"
    assert extract_project_info(text3) == "Gamma Analysis"
    assert extract_project_info(text4) is None

def test_clean_text():
    """Test text cleaning."""
    dirty_text = "  This   is   a   test   text.  \n\n  With   extra   spaces!  "
    cleaned = clean_text(dirty_text)
    
    assert cleaned == "This is a test text. With extra spaces!"

def test_truncate_text():
    """Test text truncation."""
    long_text = "This is a very long text that should be truncated at the appropriate length."
    
    truncated = truncate_text(long_text, 20)
    assert len(truncated) <= 23  # 20 + "..."
    assert truncated.endswith("...")

def test_format_file_size():
    """Test file size formatting."""
    assert format_file_size(0) == "0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    assert format_file_size(1536) == "1.5 KB"

def test_validate_customer_name():
    """Test customer name validation."""
    assert validate_customer_name("Acme Corp") == True
    assert validate_customer_name("Bio-Tech Inc") == True
    assert validate_customer_name("XYZ_Labs") == True
    assert validate_customer_name("A") == False  # Too short
    assert validate_customer_name("") == False  # Empty
    assert validate_customer_name("Invalid@Name") == False  # Invalid character

def test_validate_project_name():
    """Test project name validation."""
    assert validate_project_name("Alpha Study") == True
    assert validate_project_name("Beta-Protocol") == True
    assert validate_project_name("Gamma_Analysis") == True
    assert validate_project_name("A") == False  # Too short
    assert validate_project_name("") == False  # Empty
    assert validate_project_name("Invalid@Project") == False  # Invalid character

def test_create_metadata_from_filename():
    """Test metadata creation from filename."""
    metadata = create_metadata_from_filename("Customer_Acme_Project_Alpha_Report.pdf")
    
    assert metadata["original_filename"] == "Customer_Acme_Project_Alpha_Report.pdf"
    assert metadata["file_extension"] == ".pdf"
    assert metadata["file_basename"] == "Customer_Acme_Project_Alpha_Report"

def test_calculate_importance_score():
    """Test importance score calculation."""
    # Test with basic content
    score1 = calculate_importance_score("This is a short text.", {})
    assert 0.0 <= score1 <= 1.0
    
    # Test with important keywords
    important_text = "This is urgent and critical information requiring immediate action."
    score2 = calculate_importance_score(important_text, {})
    assert score2 > score1  # Should be higher due to keywords
    
    # Test with manual entry
    metadata = {"is_manual_entry": True}
    score3 = calculate_importance_score("Regular text", metadata)
    assert score3 > 0.5  # Should be higher due to manual entry flag
