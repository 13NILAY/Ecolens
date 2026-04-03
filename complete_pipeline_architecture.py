"""
COMPLETE PRODUCTION ESG EXTRACTION PIPELINE
============================================

Architecture:
PDF Document
     ↓
Text & Table Extraction (pdfplumber/PyMuPDF)
     ↓
Document Structuring Layer (section detection, page context)
     ↓
Text Preprocessing (cleaning, normalization)
     ↓
Candidate ESG Filter (removes non-ESG content)
     ↓
NER Model (BERT-based metric extraction)
     ↓
Metric Classifier (normalize metric names)
     ↓
Context Window Extraction (get surrounding text)
     ↓
Value Extraction (Hybrid: Regex + ML)
     ↓
Unit Normalization (standardize units)
     ↓
Confidence Scoring (aggregate all scores)
     ↓
Validation Layer (business rules, sanity checks)
     ↓
Structured JSON Output
     ↓
Logging & Feedback (optional)
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from enum import Enum


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MetricCategory(Enum):
    ENVIRONMENTAL = "Environmental"
    SOCIAL = "Social"
    GOVERNANCE = "Governance"
    UNKNOWN = "Unknown"


@dataclass
class DocumentSection:
    """Represents a section of the PDF"""
    section_id: str
    title: str
    page_number: int
    text: str
    is_table: bool = False


@dataclass
class ESGCandidate:
    """Potential ESG metric mention"""
    text: str
    start: int
    end: int
    section_id: str
    page_number: int
    confidence: float


@dataclass
class ExtractedMetric:
    """Final extracted metric with all information"""
    metric_text: str
    normalized_metric: str
    category: str
    value: Optional[float]
    unit: Optional[str]
    context: str
    page_number: int
    section_id: str
    confidence_score: float
    validation_status: str  # "VALID", "WARNING", "ERROR"
    validation_messages: List[str]


# ============================================================================
# LAYER 1: TEXT & TABLE EXTRACTION
# ============================================================================

class PDFExtractor:
    """
    Extract text and tables from PDF.
    Production: Use pdfplumber or PyMuPDF
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract(self, pdf_path: str) -> List[DocumentSection]:
        """
        Extract structured content from PDF.
        
        Returns list of DocumentSection objects.
        """
        # PRODUCTION: Implement with pdfplumber
        # This is a stub for the architecture
        
        sections = []
        
        # Example structure:
        # sections.append(DocumentSection(
        #     section_id="section_1",
        #     title="Environmental Performance",
        #     page_number=1,
        #     text="Our Scope 1 emissions were 1.2 million tCO2e...",
        #     is_table=False
        # ))
        
        return sections
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables separately for structured data"""
        # PRODUCTION: Use pdfplumber.extract_tables()
        return []


# ============================================================================
# LAYER 2: DOCUMENT STRUCTURING
# ============================================================================

class DocumentStructurer:
    """
    Structure extracted content into logical sections.
    
    - Detect section headers (Environmental, Social, Governance)
    - Maintain page context
    - Link tables to nearby text
    - Identify report metadata (year, company, reporting standard)
    """
    
    def __init__(self):
        self.section_keywords = {
            'environmental': [
                'environmental', 'climate', 'emissions', 'carbon', 'ghg',
                'energy', 'water', 'waste', 'scope 1', 'scope 2', 'scope 3'
            ],
            'social': [
                'social', 'employees', 'workforce', 'diversity', 'safety',
                'training', 'health', 'labor', 'human rights', 'community'
            ],
            'governance': [
                'governance', 'board', 'ethics', 'compliance', 'risk',
                'transparency', 'anti-corruption', 'directors'
            ]
        }
    
    def structure(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """
        Add structure metadata to sections.
        
        - Classify section category
        - Detect headers/subsections
        - Link related content
        """
        structured_sections = []
        
        for section in sections:
            # Classify section
            section_category = self._classify_section(section.text)
            
            # Add metadata (in production, modify DocumentSection to include category)
            structured_sections.append(section)
        
        return structured_sections
    
    def _classify_section(self, text: str) -> str:
        """Classify section as Environmental, Social, or Governance"""
        text_lower = text.lower()
        
        scores = {
            'environmental': 0,
            'social': 0,
            'governance': 0
        }
        
        for category, keywords in self.section_keywords.items():
            for keyword in keywords:
                scores[category] += text_lower.count(keyword)
        
        if max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def extract_metadata(self, sections: List[DocumentSection]) -> Dict:
        """Extract report metadata (year, company, standard)"""
        metadata = {
            'year': None,
            'company': None,
            'reporting_standard': []
        }
        
        # Combine first few sections for metadata extraction
        combined_text = ' '.join([s.text for s in sections[:3]])
        
        # Extract year (look for 2018-2026)
        year_match = re.search(r'\b(20[12]\d)\b', combined_text)
        if year_match:
            metadata['year'] = int(year_match.group(1))
        
        # Detect reporting standards
        standards = ['GRI', 'SASB', 'TCFD', 'CDP', 'ISO 14001']
        for standard in standards:
            if standard in combined_text:
                metadata['reporting_standard'].append(standard)
        
        return metadata


# ============================================================================
# LAYER 3: TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """
    Clean and normalize text.
    
    - Fix common PDF extraction issues
    - Normalize whitespace
    - Handle line breaks
    - Remove page headers/footers
    """
    
    def __init__(self):
        # Common PDF extraction artifacts
        self.artifacts_patterns = [
            r'\f',  # Form feed
            r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # Control characters
        ]
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove artifacts
        for pattern in self.artifacts_patterns:
            text = re.sub(pattern, '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken words (common in PDFs)
        # Example: "emis-\nsions" -> "emissions"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Chunk text for model input.
        
        CRITICAL: Use overlapping chunks to avoid splitting entities.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for period within last 100 chars
                period_pos = text.rfind('.', start, end)
                if period_pos > start + max_length - 100:
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap  # Overlap to avoid splitting entities
        
        return chunks


# ============================================================================
# LAYER 4: CANDIDATE ESG FILTER
# ============================================================================

class ESGCandidateFilter:
    """
    Filter out non-ESG content before expensive NER processing.
    
    This is a CRITICAL optimization layer:
    - Reduces false positives
    - Speeds up processing (only run NER on ESG-relevant text)
    - Improves precision
    """
    
    def __init__(self):
        # ESG indicator keywords
        self.esg_keywords = {
            # Environmental
            'emissions', 'carbon', 'co2', 'ghg', 'scope', 'energy', 'renewable',
            'water', 'waste', 'recycling', 'climate', 'environmental', 'sustainability',
            
            # Social
            'employees', 'workforce', 'diversity', 'gender', 'training', 'safety',
            'injury', 'accident', 'turnover', 'attrition', 'health', 'labor',
            
            # Governance
            'board', 'directors', 'governance', 'ethics', 'compliance', 'esg',
            'transparency', 'risk', 'audit', 'independence'
        }
        
        # Financial keywords (to exclude)
        self.financial_keywords = {
            'revenue', 'profit', 'earnings', 'ebitda', 'cash flow', 'dividend',
            'stock price', 'market cap', 'valuation', 'debt', 'equity'
        }
    
    def is_esg_candidate(self, text: str, threshold: float = 0.3) -> bool:
        """
        Determine if text likely contains ESG metrics.
        
        Returns True if text has sufficient ESG indicators.
        """
        text_lower = text.lower()
        
        # Count ESG keywords
        esg_count = sum(text_lower.count(kw) for kw in self.esg_keywords)
        
        # Count financial keywords
        financial_count = sum(text_lower.count(kw) for kw in self.financial_keywords)
        
        # Simple heuristic: ESG keywords should dominate
        total_keywords = esg_count + financial_count
        
        if total_keywords == 0:
            return False
        
        esg_ratio = esg_count / total_keywords
        
        return esg_ratio >= threshold
    
    def filter_candidates(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Filter sections to only ESG-relevant content"""
        return [s for s in sections if self.is_esg_candidate(s.text)]


# ============================================================================
# LAYER 5: NER MODEL (STUB - Production would use trained BERT)
# ============================================================================

class ESGNERModel:
    """
    Named Entity Recognition for ESG metrics.
    
    PRODUCTION: Load fine-tuned BERT/RoBERTa model
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # PRODUCTION: Load model
        # self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract ESG metric entities from text.
        
        Returns:
        [
            {
                "text": "Scope 1 emissions",
                "start": 10,
                "end": 27,
                "label": "GHG_SCOPE_1",
                "confidence": 0.95
            }
        ]
        """
        # PRODUCTION: Run NER model inference
        # For now, return stub
        return []


# ============================================================================
# LAYER 6: METRIC CLASSIFIER
# ============================================================================

class MetricClassifier:
    """
    Classify extracted metrics into normalized labels.
    
    Maps: "direct emissions" -> "GHG_SCOPE_1"
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # PRODUCTION: Load classification model
    
    def classify(self, metric_text: str, context: str = "") -> Tuple[str, float]:
        """
        Classify metric text to normalized label.
        
        Returns: (normalized_metric, confidence)
        """
        # PRODUCTION: Run classifier
        return "UNKNOWN", 0.0


# ============================================================================
# LAYER 7: CONTEXT WINDOW EXTRACTION
# ============================================================================

class ContextExtractor:
    """
    Extract relevant context around detected metrics.
    
    This helps with value extraction and validation.
    """
    
    def extract_context(
        self, 
        text: str, 
        entity_start: int, 
        entity_end: int,
        window_size: int = 150
    ) -> str:
        """
        Extract context window around entity.
        
        Includes ±window_size characters around the entity.
        """
        context_start = max(0, entity_start - window_size)
        context_end = min(len(text), entity_end + window_size)
        
        return text[context_start:context_end]


# ============================================================================
# LAYER 8: VALUE EXTRACTION (HYBRID)
# ============================================================================

class ValueExtractor:
    """
    Extract numeric values and units.
    
    HYBRID APPROACH:
    1. Regex patterns (fast, handles 80% of cases)
    2. ML model fallback (for complex cases)
    """
    
    def __init__(self):
        # Common number patterns
        self.number_patterns = [
            r'([\d,]+\.?\d*)\s*([a-zA-Z/%]+)',  # "1,200.5 tonnes"
            r'([\d,]+\.?\d*)',  # Just number
        ]
        
        # Unit variations
        self.unit_mappings = {
            'tonnes': 'tCO2e',
            'tons': 'tCO2e',
            'tco2e': 'tCO2e',
            'mtco2e': 'Mt CO2e',
            'million tonnes': 'Mt CO2e',
            'gwh': 'GWh',
            'mwh': 'MWh',
            'm3': 'm³',
            'cubic meters': 'm³',
        }
    
    def extract(self, context: str) -> List[Dict]:
        """
        Extract (value, unit) pairs from context.
        
        Returns:
        [
            {"value": 1200.5, "unit": "tCO2e", "confidence": 0.9}
        ]
        """
        results = []
        
        for pattern in self.number_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            
            for match in matches:
                value_str = match.group(1).replace(',', '')
                
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                
                # Extract unit if present
                unit = ""
                if len(match.groups()) > 1:
                    unit = match.group(2)
                    unit = self._normalize_unit(unit)
                
                results.append({
                    "value": value,
                    "unit": unit,
                    "confidence": 0.85,  # Regex confidence
                    "extraction_method": "regex"
                })
        
        return results
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit to standard form"""
        unit_lower = unit.lower().strip()
        return self.unit_mappings.get(unit_lower, unit)


# ============================================================================
# LAYER 9: UNIT NORMALIZATION
# ============================================================================

class UnitNormalizer:
    """
    Standardize units to canonical forms.
    
    - Convert between units (e.g., tonnes -> kg)
    - Standardize representations
    - Handle abbreviations
    """
    
    def __init__(self):
        self.conversion_factors = {
            ('tonnes', 'kg'): 1000,
            ('Mt', 'tonnes'): 1_000_000,
            ('GWh', 'MWh'): 1000,
        }
    
    def normalize(self, value: float, unit: str) -> Tuple[float, str]:
        """Normalize value and unit to standard form"""
        # Simple normalization (production would be more comprehensive)
        return value, unit
    
    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between units"""
        key = (from_unit, to_unit)
        if key in self.conversion_factors:
            return value * self.conversion_factors[key]
        return value


# ============================================================================
# LAYER 10: CONFIDENCE SCORING
# ============================================================================

class ConfidenceScorer:
    """
    Aggregate confidence scores from all pipeline components.
    
    Combines:
    - NER confidence
    - Classification confidence
    - Value extraction confidence
    - Context quality score
    """
    
    def calculate(
        self,
        ner_confidence: float,
        classification_confidence: float,
        value_confidence: float,
        has_unit: bool
    ) -> float:
        """
        Calculate aggregate confidence score.
        
        Weighted average with bonuses/penalties.
        """
        # Base weighted average
        weights = [0.4, 0.3, 0.3]  # NER, Classification, Value
        base_score = (
            ner_confidence * weights[0] +
            classification_confidence * weights[1] +
            value_confidence * weights[2]
        )
        
        # Bonus for having unit
        if has_unit:
            base_score = min(1.0, base_score * 1.1)
        
        return base_score


# ============================================================================
# LAYER 11: VALIDATION LAYER
# ============================================================================

class MetricValidator:
    """
    Validate extracted metrics using business rules.
    
    Checks:
    - Value ranges (e.g., percentages 0-100)
    - Required fields
    - Logical consistency
    - Outlier detection
    """
    
    def __init__(self):
        # Define acceptable ranges per metric
        self.value_ranges = {
            'GHG_SCOPE_1': (0, 100_000_000),
            'GHG_SCOPE_2': (0, 100_000_000),
            'GHG_SCOPE_3': (0, 500_000_000),
            'RENEWABLE_ENERGY_PCT': (0, 100),
            'GENDER_DIVERSITY_PCT': (0, 100),
            'WASTE_RECYCLED_PCT': (0, 100),
        }
    
    def validate(self, metric: ExtractedMetric) -> Tuple[str, List[str]]:
        """
        Validate extracted metric.
        
        Returns: (status, messages)
        status: "VALID", "WARNING", "ERROR"
        """
        messages = []
        status = "VALID"
        
        # Check required fields
        if not metric.metric_text:
            messages.append("Missing metric text")
            status = "ERROR"
        
        if metric.value is None:
            messages.append("Missing value")
            status = "WARNING"
        
        # Check value range
        if metric.value is not None and metric.normalized_metric in self.value_ranges:
            min_val, max_val = self.value_ranges[metric.normalized_metric]
            
            if metric.value < min_val or metric.value > max_val:
                messages.append(f"Value {metric.value} outside expected range [{min_val}, {max_val}]")
                status = "WARNING"
        
        # Check confidence
        if metric.confidence_score < 0.5:
            messages.append(f"Low confidence: {metric.confidence_score:.2f}")
            if status == "VALID":
                status = "WARNING"
        
        return status, messages


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class ESGExtractionPipeline:
    """
    Complete end-to-end ESG extraction pipeline.
    """
    
    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None
    ):
        # Initialize all components
        self.pdf_extractor = PDFExtractor()
        self.structurer = DocumentStructurer()
        self.preprocessor = TextPreprocessor()
        self.candidate_filter = ESGCandidateFilter()
        self.ner_model = ESGNERModel(ner_model_path)
        self.classifier = MetricClassifier(classifier_model_path)
        self.context_extractor = ContextExtractor()
        self.value_extractor = ValueExtractor()
        self.unit_normalizer = UnitNormalizer()
        self.confidence_scorer = ConfidenceScorer()
        self.validator = MetricValidator()
    
    def process(self, pdf_path: str) -> Dict:
        """
        Process a PDF and extract ESG metrics.
        
        Returns structured JSON with all extracted metrics.
        """
        results = {
            'metadata': {},
            'metrics': [],
            'warnings': [],
            'processing_log': []
        }
        
        # Layer 1: Extract text & tables
        sections = self.pdf_extractor.extract(pdf_path)
        results['processing_log'].append(f"Extracted {len(sections)} sections")
        
        # Layer 2: Structure document
        sections = self.structurer.structure(sections)
        results['metadata'] = self.structurer.extract_metadata(sections)
        
        # Layer 3: Preprocess text
        for section in sections:
            section.text = self.preprocessor.preprocess(section.text)
        
        # Layer 4: Filter ESG candidates
        esg_sections = self.candidate_filter.filter_candidates(sections)
        results['processing_log'].append(
            f"Filtered to {len(esg_sections)} ESG-relevant sections"
        )
        
        # Layers 5-11: Process each section
        for section in esg_sections:
            section_metrics = self._process_section(section)
            results['metrics'].extend(section_metrics)
        
        # Final validation
        for metric in results['metrics']:
            if metric['validation_status'] != 'VALID':
                results['warnings'].append({
                    'metric': metric['normalized_metric'],
                    'messages': metric['validation_messages']
                })
        
        return results
    
    def _process_section(self, section: DocumentSection) -> List[Dict]:
        """Process a single section through layers 5-11"""
        metrics = []
        
        # Layer 5: NER - Extract entities
        entities = self.ner_model.extract_entities(section.text)
        
        for entity in entities:
            # Layer 6: Classify metric
            normalized_metric, class_conf = self.classifier.classify(
                entity['text'], 
                section.text
            )
            
            # Layer 7: Extract context
            context = self.context_extractor.extract_context(
                section.text,
                entity['start'],
                entity['end']
            )
            
            # Layer 8: Extract value
            values = self.value_extractor.extract(context)
            
            if not values:
                continue
            
            value_data = values[0]  # Take first/best match
            
            # Layer 9: Normalize unit
            normalized_value, normalized_unit = self.unit_normalizer.normalize(
                value_data['value'],
                value_data['unit']
            )
            
            # Layer 10: Calculate confidence
            confidence = self.confidence_scorer.calculate(
                ner_confidence=entity['confidence'],
                classification_confidence=class_conf,
                value_confidence=value_data['confidence'],
                has_unit=bool(normalized_unit)
            )
            
            # Create metric object
            metric = ExtractedMetric(
                metric_text=entity['text'],
                normalized_metric=normalized_metric,
                category="Unknown",  # Would be set by classifier
                value=normalized_value,
                unit=normalized_unit,
                context=context,
                page_number=section.page_number,
                section_id=section.section_id,
                confidence_score=confidence,
                validation_status="PENDING",
                validation_messages=[]
            )
            
            # Layer 11: Validate
            status, messages = self.validator.validate(metric)
            metric.validation_status = status
            metric.validation_messages = messages
            
            metrics.append(asdict(metric))
        
        return metrics


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of complete pipeline"""
    
    # Initialize pipeline
    pipeline = ESGExtractionPipeline(
        ner_model_path='./models/ner_model',
        classifier_model_path='./models/classifier_model'
    )
    
    # Process PDF
    results = pipeline.process('example_esg_report.pdf')
    
    # Save results
    with open('extracted_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Extracted {len(results['metrics'])} metrics")
    print(f"Warnings: {len(results['warnings'])}")


if __name__ == "__main__":
    main()
