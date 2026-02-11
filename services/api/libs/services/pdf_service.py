import pdfplumber
from fastapi import UploadFile
from loguru import logger
import re


def extract_text_from_pdf(file: UploadFile) -> str:
    text = ""
    
    try:
        with pdfplumber.open(file.file) as pdf:
            logger.info(f"Extraindo texto de {file.filename} ({len(pdf.pages)} páginas)")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    
                    if page_text:
                        cleaned_page = _clean_extracted_text(page_text)
                        
                        if cleaned_page:
                            text += cleaned_page + "\n\n"
                            
                except Exception as e:
                    logger.warning(f"Erro ao extrair página {page_num}: {e}")
                    continue
            
            logger.info(f"Texto extraído: {len(text)} caracteres de {file.filename}")
            
    except Exception as e:
        logger.error(f"Erro ao processar PDF {file.filename}: {e}")
        raise
    
    return text.strip()


def _clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    text = re.sub(r'\(cid:\d+\)', '', text)  
    text = re.sub(r'[•]{3,}', '', text)  
    text = re.sub(r'[\.]{4,}', '...', text)  
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        if line.isdigit():
            continue
        
        if len(line) < 3:
            continue
        
        if re.match(r'^[\s\W]+$', line):
            continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()