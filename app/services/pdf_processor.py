import os
import re
import hashlib
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class PageContent(BaseModel):
    page_num: int
    text: str
    word_count: int
    ocr_used: bool

class PDFProcessor:
    """
    Handles PDF text extraction with multi-stage logic:
    1. pdfplumber (high fidelity text/tables)
    2. fitz/PyMuPDF (backup for speed/repair)
    3. Tesseract OCR (fallback for scanned pages)
    """

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Compute SHA-256 for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and remove null bytes."""
        if not text:
            return ""
        # Remove null bytes
        text = text.replace('\x00', '')
        # Collapse multiple whitespaces/newlines
        text = re.sub(r'\s{3,}', '  ', text)
        return text.strip()

    def process_pdf(self, file_path: str) -> List[PageContent]:
        """Extract text from PDF page by page."""
        pages_content = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    ocr_used = False
                    
                    # Detect scanned page (less than 50 chars)
                    if not text or len(text.strip()) < 50:
                        logger.info(f"Page {i+1} appears scanned. Falling back to OCR.")
                        # Convert PDF page to image for OCR
                        # Using fitz for faster page-to-pixmap conversion
                        doc = fitz.open(file_path)
                        fitz_page = doc.load_page(i)
                        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher DPI
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img)
                        ocr_used = True
                        doc.close()

                    clean_txt = self.clean_text(text)
                    pages_content.append(PageContent(
                        page_num=i + 1,
                        text=clean_txt,
                        word_count=len(clean_txt.split()),
                        ocr_used=ocr_used
                    ))
            
            return pages_content
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

pdf_processor = PDFProcessor()
