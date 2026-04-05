"""
COMPLETE PRODUCTION ESG EXTRACTION PIPELINE (v3 — Delta Precision Fixes)
============================================================================

Architecture:
PDF Document
     ↓
Text & Table Extraction (pdfplumber/PyMuPDF)
     ↓
Document Structuring Layer (section detection, page context)
     ↓
Text Preprocessing (cleaning, normalization)
     ↓
Candidate ESG Filter (v3: 2-keyword threshold + confidence >= 0.4)
     ↓
NER Model (BERT-based metric extraction)
     ↓
Metric Classifier (improved input: [SECTION:] METRIC: CONTEXT:)
     ↓
Context Window Extraction (sentence-based, ±100 char fallback)
     ↓
Value Extraction (v3: temporal pre-filter + hard proximity <=100 + closest only)
     ├── Unit Whitelist Validation
     ├── Invalid Pattern Filtering (years, page numbers, broken notation)
     ├── Noise Value Rejection (integers <5 or in {1,2,3,4,5,10} unless %)
     └── Hard Proximity Constraint (>100 chars from entity -> discard)
     ↓
Unit Normalization (standardize units via lookup table)
     ↓
Validation Layer (hard rules, metric-unit compat, keyword checks)
     ├── Score Metric Guard (range 0-100, min 20, requires score keyword)
     └── Strict Keyword Validation (0 matches=discard, 1=0.7 penalty)
     ↓
Confidence Scoring (v3: min-penalty strategy, score penalty, clamp [0.3, 0.95])
     ↓
Fuzzy Deduplication (group by metric+unit, +/-5% value clustering)
     ↓
Structured JSON Output

v3 Delta Fixes (on top of v2 — ALL rule-based, NO new ML):
  FIX 1: Score metric strict validation (range 0-100, min 20, keyword required)
  FIX 2: Confidence anti-stacking (min optional penalty, not multiplicative)
  FIX 3: Value selection — temporal pre-filter + hard proximity <=100 + closest
  FIX 4: ESG filter tightened (2 keywords AND confidence >= 0.4)
  FIX 5: Strict keyword validation (0 matches = hard discard)
  FIX 6: Score metric confidence penalty (x0.7)
  FIX 7: Hard proximity discard (>100 chars from entity)
  FIX 8: Noise value rejection (integers <5, standalone {1-5,10} unless %)
  FIX 9: Confidence clamp enforced at [0.3, 0.95]
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from enum import Enum
from collections import defaultdict


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
    validation_status: str  # "VALID", "WARNING", "INVALID"
    validation_messages: List[str]


# ============================================================================
# CONSTANTS — Unit Whitelist, Metric-Unit Map, Keyword Map
# ============================================================================

VALID_UNITS: Set[str] = {
    "tCO2e", "Mt CO2e",
    "%",
    "GWh", "MWh", "kWh", "Mn kWh",
    "MJ", "GJ", "TJ",
    "m³", "m3", "KL", "kilolitres", "liters", "litres",
    "employees", "FTE",
    "hours", "hours/employee",
    "kg", "tonnes", "t", "MT",
    "crore", "lakh",
}

UNIT_NORMALIZATION: Dict[str, str] = {
    'tonnes': 'tCO2e', 'tons': 'tCO2e', 'tco2e': 'tCO2e',
    'mtco2e': 'Mt CO2e', 'million tonnes': 'Mt CO2e',
    'gwh': 'GWh', 'mwh': 'MWh', 'kwh': 'kWh',
    'm3': 'm³', 'cubic meters': 'm³', 'cubic metres': 'm³',
    '%': '%', 'percent': '%', 'percentage': '%',
    'fte': 'FTE', 'employees': 'employees',
    'hours': 'hours', 'hours/employee': 'hours/employee',
    'mt': 'MT', 'metric tonnes': 'MT',
    'kl': 'KL', 'kilolitres': 'KL', 'kiloliters': 'KL',
    'mj': 'MJ', 'gj': 'GJ', 'tj': 'TJ',
    'mn kwh': 'Mn kWh',
    'crore': 'crore', 'lakh': 'lakh',
    'kg': 'kg', 't': 't',
}

METRIC_UNIT_MAP: Dict[str, Set[str]] = {
    'SCOPE_1': {'tCO2e', 'Mt CO2e', 'tonnes', 't', 'MT', 'kg'},
    'SCOPE_2': {'tCO2e', 'Mt CO2e', 'tonnes', 't', 'MT', 'kg'},
    'SCOPE_3': {'tCO2e', 'Mt CO2e', 'tonnes', 't', 'MT', 'kg'},
    'ENERGY_CONSUMPTION': {'GWh', 'MWh', 'kWh', 'MJ', 'GJ', 'TJ', 'Mn kWh', 'crore'},
    'WATER_USAGE': {'m³', 'KL', 'kilolitres', 'liters', 'litres', 'crore'},
    'WASTE_GENERATED': {'tonnes', 'MT', 't', 'kg'},
    'ESG_SCORE': set(),  # unitless allowed
}

METRIC_KEYWORDS: Dict[str, List[str]] = {
    'SCOPE_1': ['emissions', 'co2', 'ghg', 'carbon', 'scope 1', 'scope1', 'direct emissions'],
    'SCOPE_2': ['emissions', 'co2', 'ghg', 'carbon', 'scope 2', 'scope2', 'indirect emissions'],
    'SCOPE_3': ['emissions', 'co2', 'ghg', 'carbon', 'scope 3', 'scope3', 'value chain'],
    'ENERGY_CONSUMPTION': ['energy', 'electricity', 'power', 'consumption', 'kwh', 'mwh', 'gwh'],
    'WATER_USAGE': ['water', 'withdrawal', 'discharge', 'consumption'],
    'WASTE_GENERATED': ['waste', 'generated', 'disposal', 'solid waste'],
    'ESG_SCORE': ['esg', 'score', 'rating', 'sustainability', 'index'],
}

UNITLESS_ALLOWED_METRICS: Set[str] = {
    'ESG_SCORE',
}

TEMPORAL_KEYWORDS: List[str] = [
    'compared', 'previous', 'last year', 'prior year', 'preceding',
    'increase', 'decrease', 'growth', 'decline', 'change',
]

INVALID_UNIT_WORDS: Set[str] = {
    'the', 'a', 'an', 'and', 'or', 'for', 'of', 'in', 'to', 'by', 'on', 'at',
    'is', 'it', 'as', 'be', 'was', 'has', 'had', 'are', 'its', 'our', 'we',
    'all', 'key', 'no', 'yes', 'na', 'nil',
    'total', 'from', 'note', 'business', 'gri', 'fy', 'dry', 'bio',
    'male', 'female',
    'accounts', 'carbon', 'emissions', 'turnover',
    'worked', 'compared', 'standards', 'awareness',
    'e', 'p', 'g', 'r', 's',
}

GENERIC_ENTITY_WORDS: Set[str] = {
    'total', 'number', 'energy', 'water', 'waste', 'carbon', 'scope',
    'paid', 'value', 'women', 'employee', 'os', 'km',
}


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
        """Extract structured content from PDF."""
        # PRODUCTION: Implement with pdfplumber
        sections = []
        return sections
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables separately for structured data"""
        return []


# ============================================================================
# LAYER 2: DOCUMENT STRUCTURING
# ============================================================================

class DocumentStructurer:
    """
    Structure extracted content into logical sections.
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
        """Add structure metadata to sections."""
        structured_sections = []
        for section in sections:
            section_category = self._classify_section(section.text)
            structured_sections.append(section)
        return structured_sections
    
    def _classify_section(self, text: str) -> str:
        """Classify section as Environmental, Social, or Governance"""
        text_lower = text.lower()
        scores = {cat: 0 for cat in self.section_keywords}
        for category, keywords in self.section_keywords.items():
            for keyword in keywords:
                scores[category] += text_lower.count(keyword)
        if max(scores.values()) == 0:
            return 'unknown'
        return max(scores, key=scores.get)
    
    def extract_metadata(self, sections: List[DocumentSection]) -> Dict:
        """Extract report metadata (year, company, standard)"""
        metadata = {'year': None, 'company': None, 'reporting_standard': []}
        combined_text = ' '.join([s.text for s in sections[:3]])
        year_match = re.search(r'\b(20[12]\d)\b', combined_text)
        if year_match:
            metadata['year'] = int(year_match.group(1))
        standards = ['GRI', 'SASB', 'TCFD', 'CDP', 'ISO 14001']
        for standard in standards:
            if standard in combined_text:
                metadata['reporting_standard'].append(standard)
        return metadata


# ============================================================================
# LAYER 3: TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Clean and normalize text for downstream processing."""
    
    def __init__(self):
        self.artifacts_patterns = [
            r'\f',
            r'[\x00-\x08\x0b\x0c\x0e-\x1f]',
        ]
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize text"""
        for pattern in self.artifacts_patterns:
            text = re.sub(pattern, '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """Chunk text for model input with sentence-boundary awareness."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            if end < len(text):
                period_pos = text.rfind('.', start, end)
                if period_pos > start + max_length - 100:
                    end = period_pos + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks


# ============================================================================
# LAYER 4: CANDIDATE ESG FILTER (FIX 9 — Balanced)
# ============================================================================

class ESGCandidateFilter:
    """
    Filter out non-ESG content before expensive NER processing.
    
    FIX 9: Balanced filtering
    - 2-keyword threshold (NOT 3)
    - No numeric requirement
    - Numeric presence boosts confidence
    - Negative keyword deprioritization
    """
    
    def __init__(self):
        self.esg_keywords = {
            'emissions', 'carbon', 'co2', 'ghg', 'scope', 'energy', 'renewable',
            'water', 'waste', 'recycling', 'climate', 'environmental', 'sustainability',
            'employees', 'workforce', 'diversity', 'gender', 'training', 'safety',
            'injury', 'accident', 'turnover', 'attrition', 'health', 'labor',
            'board', 'directors', 'governance', 'ethics', 'compliance', 'esg',
            'transparency', 'risk', 'audit', 'independence'
        }
        self.negative_keywords = {
            'revenue', 'profit', 'earnings', 'ebitda', 'cash flow', 'dividend',
            'stock price', 'market cap', 'valuation', 'debt', 'equity',
            'litigation', 'lawsuit', 'arbitration',
        }
    
    def is_esg_candidate(self, text: str) -> bool:
        """Determine if text likely contains ESG metrics.

        FIX 4 (v3): Require keyword_count >= 2 AND ESG confidence >= 0.4
        to reduce the 95% classification rate and filter noise.
        """
        text_lower = text.lower()
        esg_count = sum(1 for kw in self.esg_keywords if kw in text_lower)
        if esg_count < 2:
            return False
        confidence = self.get_esg_confidence(text)
        return confidence >= 0.4
    
    def get_esg_confidence(self, text: str) -> float:
        """Get ESG relevance confidence score."""
        text_lower = text.lower()
        esg_count = sum(1 for kw in self.esg_keywords if kw in text_lower)
        neg_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        confidence = min(esg_count / 5.0, 1.0)
        if neg_count > 0:
            confidence *= max(0.5, 1.0 - neg_count * 0.15)
        
        # Numeric presence boosts confidence
        if re.search(r'\d+\.?\d*', text):
            confidence = min(1.0, confidence + 0.2)
        
        return confidence
    
    def filter_candidates(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Filter sections to only ESG-relevant content"""
        return [s for s in sections if self.is_esg_candidate(s.text)]


# ============================================================================
# LAYER 5: NER MODEL (STUB)
# ============================================================================

class ESGNERModel:
    """Named Entity Recognition for ESG metrics."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract ESG metric entities from text."""
        # PRODUCTION: Run NER model inference
        return []


# ============================================================================
# LAYER 6: METRIC CLASSIFIER (FIX 6 — Improved Input Format)
# ============================================================================

class MetricClassifier:
    """
    Classify extracted metrics into normalized labels.
    
    FIX 6: Improved input format for better disambiguation.
    Format: "[SECTION: {section_type}] METRIC: {metric_text} CONTEXT: {context}"
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
    
    def classify(self, metric_text: str, context: str = "", section_type: str = "") -> Tuple[str, float]:
        """
        Classify metric text to normalized label.
        
        Input format: "[SECTION: Environmental] METRIC: Scope 2 emissions CONTEXT: ..."
        """
        input_text = (
            f"[SECTION: {section_type}] "
            f"METRIC: {metric_text} "
            f"CONTEXT: {context}"
        )
        # PRODUCTION: Run classifier model with input_text
        return "UNKNOWN", 0.0


# ============================================================================
# LAYER 7: CONTEXT WINDOW EXTRACTION (FIX 5 — Sentence-Based)
# ============================================================================

class ContextExtractor:
    """
    Extract relevant context around detected metrics.
    
    FIX 5: Sentence-based extraction with character-window fallback.
    """
    
    def extract_context(
        self, 
        text: str, 
        entity_start: int, 
        entity_end: int,
        window_size: int = 100
    ) -> str:
        """
        Extract sentence-based context around entity.
        Falls back to ±window_size characters.
        """
        # Try sentence-based
        sentence_ctx = self._get_sentence_context(text, entity_start, entity_end)
        if sentence_ctx and len(sentence_ctx) >= 30:
            return sentence_ctx
        
        # Fallback
        start = max(0, entity_start - window_size)
        end = min(len(text), entity_end + window_size)
        return text[start:end]
    
    def _get_sentence_context(self, text: str, entity_start: int, entity_end: int) -> Optional[str]:
        """Extract sentence(s) containing entity span + next sentence."""
        boundaries = list(re.finditer(r'[.!?]\s+', text))
        if not boundaries:
            return None
        
        sentences = []
        prev_end = 0
        for match in boundaries:
            sentences.append((prev_end, match.end()))
            prev_end = match.end()
        if prev_end < len(text):
            sentences.append((prev_end, len(text)))
        
        entity_idx = None
        for i, (s, e) in enumerate(sentences):
            if s <= entity_start < e:
                entity_idx = i
                break
        
        if entity_idx is None:
            return None
        
        start = sentences[entity_idx][0]
        end_idx = min(entity_idx + 1, len(sentences) - 1)
        end = sentences[end_idx][1]
        return text[start:end].strip()


# ============================================================================
# LAYER 8: VALUE EXTRACTION (FIXES 1, 2, 3, 3b)
# ============================================================================

class ValueExtractor:
    """
    Extract numeric values and units with quality gates.
    
    FIX 1: Entity-proximity extraction (±50 chars, closest value)
    FIX 2: Strict unit whitelist validation
    FIX 3: Invalid numeric pattern filtering
    FIX 3b: Temporal keyword awareness
    """
    
    def __init__(self):
        self.number_patterns = [
            (r'([\d,]+\.?\d*)\s*(%)', 'pct'),
            (r'([\d,]+\.\d+)\s+([a-zA-Z/%³]+(?:\s+[a-zA-Z]+)?)', 'full_float'),
            (r'([\d,]+)\s+([a-zA-Z/%³]+(?:\s+[a-zA-Z]+)?)', 'full_int'),
            (r'([\d,]+\.\d+)', 'float_only'),
            (r'([\d,]+)', 'int_only'),
        ]
        
        self.confidence_map = {
            'pct': 0.95, 'full_float': 0.90, 'full_int': 0.85,
            'float_only': 0.60, 'int_only': 0.50,
        }
    
    def extract(
        self,
        context: str,
        entity_start: Optional[int] = None,
        entity_end: Optional[int] = None,
    ) -> List[Dict]:
        """
        Extract (value, unit) pairs with quality filtering.
        
        Returns list sorted by proximity to entity (if positions given),
        with invalid patterns already filtered out.
        """
        candidates = []
        seen_spans = set()
        
        for pattern, ptype in self.number_patterns:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                span = (match.start(), match.end())
                if any(self._overlaps(span, s) for s in seen_spans):
                    continue
                
                try:
                    value = float(match.group(1).replace(',', ''))
                except ValueError:
                    continue
                
                unit = ""
                if len(match.groups()) >= 2 and match.group(2):
                    unit = self._normalize_unit(match.group(2).strip())
                
                # FIX 2: Reject invalid units
                if unit and not self._is_valid_unit(unit):
                    continue
                
                # FIX 3: Reject invalid patterns
                if not self._is_valid_value(value, match.start(), context):
                    continue

                # FIX 8 (v3): Reject small integers unless percentage
                if value < 5 and unit != "%":
                    continue
                # FIX 8 (v3): Reject common standalone noise integers
                if value in {1, 2, 3, 4, 5, 10} and unit != "%":
                    continue
                
                candidates.append({
                    'value': value,
                    'unit': unit,
                    'confidence': self.confidence_map[ptype],
                    'match_start': match.start(),
                    'match_end': match.end(),
                })
                seen_spans.add(span)
        
        if not candidates:
            return []
        
        # FIX 1: Sort by proximity to entity
        if entity_start is not None and entity_end is not None:
            entity_center = (entity_start + entity_end) / 2

            # FIX 3 (v3) Step 1: Remove candidates preceded by temporal keywords (within 50 chars)
            candidates = [
                c for c in candidates
                if not any(
                    kw in context[max(0, c['match_start'] - 50):c['match_start']].lower()
                    for kw in TEMPORAL_KEYWORDS
                )
            ]

            # FIX 3 (v3) / FIX 7 (v3) Step 2: Hard proximity constraint — discard if > 100 chars away
            candidates = [
                c for c in candidates
                if abs((c['match_start'] + c['match_end']) / 2 - entity_center) <= 100
            ]

            if not candidates:
                return []

            # FIX 3 (v3) Step 3: Select ONLY the closest candidate
            best = min(candidates, key=lambda c: abs((c['match_start'] + c['match_end']) / 2 - entity_center))
            return [best]

            # (legacy proximity scoring below — replaced by closest-only logic above)
            for c in candidates:
                mc = (c['match_start'] + c['match_end']) / 2
                dist = abs(mc - entity_center)
                c['proximity'] = 1.0 if dist <= 50 else (0.7 if dist <= 100 else 0.4)
                
                # FIX 3b: Penalize values after temporal keywords
                before = context[max(0, c['match_start'] - 60):c['match_start']].lower()
                if any(kw in before for kw in TEMPORAL_KEYWORDS):
                    c['proximity'] *= 0.5
            
            candidates.sort(key=lambda c: -c['proximity'])
        
        return candidates
    
    def _normalize_unit(self, unit_raw: str) -> str:
        return UNIT_NORMALIZATION.get(unit_raw.lower().strip(), unit_raw)
    
    def _is_valid_unit(self, unit: str) -> bool:
        if unit.lower() in INVALID_UNIT_WORDS:
            return False
        if len(unit) <= 1 and unit != '%':
            return False
        if unit in VALID_UNITS:
            return True
        normalized = UNIT_NORMALIZATION.get(unit.lower(), '')
        return normalized in VALID_UNITS
    
    def _is_valid_value(self, value: float, match_start: int, context: str) -> bool:
        """FIX 3: Reject years, page numbers, reference numbers."""
        if 1900 <= value <= 2030 and value == int(value):
            return False
        nearby = context[max(0, match_start - 30):match_start].lower()
        if value < 300 and value == int(value):
            if any(kw in nearby for kw in ['page', 'p.', 'section', 'principle', 'gri ']):
                return False
        if re.search(r'(?:gri|iso|p\d|e\d)\s*$', nearby, re.IGNORECASE):
            return False
        if re.search(r'fy\s*$', nearby, re.IGNORECASE) and value < 100:
            return False
        return True
    
    @staticmethod
    def _overlaps(s1, s2):
        return s1[0] < s2[1] and s2[0] < s1[1]


# ============================================================================
# LAYER 9: UNIT NORMALIZATION
# ============================================================================

class UnitNormalizer:
    """Standardize units to canonical forms using lookup table."""
    
    def normalize(self, value: float, unit: str) -> Tuple[float, str]:
        normalized = UNIT_NORMALIZATION.get(unit.lower(), unit)
        return value, normalized


# ============================================================================
# LAYER 10: CONFIDENCE SCORING (FIX 8 — Penalties + Clamping)
# ============================================================================

class ConfidenceScorer:
    """
    Aggregate confidence scores with realistic penalties.
    
    FIX 8: Weighted average + cumulative penalties + hard clamp [0.3, 0.95].
    """
    
    def calculate(
        self,
        ner_confidence: float,
        classification_confidence: float,
        value_confidence: float,
        unit: str,
        entity_text: str,
        num_candidates: int = 1,
        validation_penalty: float = 1.0,
        keyword_match: bool = True,
        metric: str = "",
    ) -> float:
        """Calculate aggregate confidence with penalties.

        FIX 2 (v3): Apply only the STRONGEST optional penalty (not all stacked)
        to prevent over-penalization. Target avg confidence: 55-75%.
        FIX 6 (v3): Score metric dominance penalty.
        FIX 9 (v3): Hard clamp [0.3, 0.95].
        """
        # Base weighted average
        base = (
            ner_confidence * 0.4 +
            classification_confidence * 0.3 +
            value_confidence * 0.3
        )

        # Hard validation penalty always applied
        base *= validation_penalty

        # Collect optional penalties — apply only the strongest (min value)
        optional_penalties = []
        if not unit:
            optional_penalties.append(0.7)
        if num_candidates > 1:
            optional_penalties.append(0.6)
        if len(entity_text.strip()) < 3:
            optional_penalties.append(0.6)
        if entity_text.strip().lower() in GENERIC_ENTITY_WORDS:
            optional_penalties.append(0.7)
        if not keyword_match:
            optional_penalties.append(0.5)

        if optional_penalties:
            base *= min(optional_penalties)

        # FIX 6 (v3): Reduce score metric dominance
        if metric.endswith("_SCORE"):
            base *= 0.7

        # FIX 9 (v3): Clamp to realistic range
        return round(max(0.3, min(0.95, base)), 4)


# ============================================================================
# LAYER 11: VALIDATION (FIX 7 — Hard Rules + Compatibility + Keywords)
# ============================================================================

class MetricValidator:
    """
    Validate extracted metrics with strict domain rules.
    
    FIX 7:  Range validation
    FIX 7b: Year/page noise filtering  
    FIX 7c: Metric-unit compatibility
    FIX 7d: Metric-specific keyword validation
    FIX 7e: Unitless metric rules
    """
    
    def __init__(self):
        self.value_ranges = {
            'SCOPE_1': (0, 100_000_000),
            'SCOPE_2': (0, 100_000_000),
            'SCOPE_3': (0, 500_000_000),
            'ENERGY_CONSUMPTION': (0, 50_000_000),
            'WATER_USAGE': (0, 100_000_000),
            'WASTE_GENERATED': (0, 10_000_000),
            'ESG_SCORE': (0, 100),
        }
    
    def validate(
        self,
        metric: str,
        value: float,
        unit: str,
        context: str,
        entity_text: str,
    ) -> Tuple[str, List[str], float]:
        """
        Validate metric. Returns (status, issues, confidence_penalty_multiplier).
        penalty=0.0 means discard.
        """
        issues = []
        status = "VALID"
        penalty = 1.0

        # FIX 1 (v3): Strict score metric validation — discard fake/low scores
        if metric.endswith("_SCORE"):
            if not (0 <= value <= 100):
                issues.append(f"Score value {value} out of valid range [0,100]")
                return "INVALID", issues, 0.0
            if value < 20:
                issues.append(f"Score value {value} too small (likely table index)")
                return "INVALID", issues, 0.0
            if not any(kw in context.lower() for kw in ["score", "rating", "index"]):
                issues.append(f"No score-related keyword in context for {metric}")
                return "INVALID", issues, 0.0

        # Range validation
        if metric in self.value_ranges:
            mn, mx = self.value_ranges[metric]
            if value < mn:
                issues.append(f"Value {value} below min {mn}")
                status = "INVALID"
                penalty *= 0.5
            if value > mx:
                issues.append(f"Value {value} above max {mx}")
                status = "INVALID"
                penalty *= 0.5
        
        # ESG Score range check
        if metric == 'ESG_SCORE' and not (0 <= value <= 100):
            issues.append(f"ESG Score {value} out of range [0, 100]")
            status = "INVALID"
            penalty *= 0.3
        
        # FIX 7c: Metric-unit compatibility
        if metric in METRIC_UNIT_MAP and METRIC_UNIT_MAP[metric]:
            if unit and unit not in METRIC_UNIT_MAP[metric]:
                issues.append(f"Unit '{unit}' incompatible with '{metric}'")
                status = "INVALID"
                penalty *= 0.0
        
        # FIX 7e: Unitless rules
        if not unit and metric not in UNITLESS_ALLOWED_METRICS:
            issues.append(f"Missing unit for '{metric}'")
            status = "INVALID"
            penalty *= 0.0
        
        # FIX 7d: Keyword validation
        kw_penalty = self._check_keywords(metric, context, entity_text)
        if kw_penalty < 1.0:
            if kw_penalty == 0.0:
                issues.append(f"No domain keywords for '{metric}'")
                status = "INVALID"
            else:
                issues.append(f"Weak keyword match for '{metric}'")
                if status == "VALID":
                    status = "WARNING"
            penalty *= kw_penalty
        
        return status, issues, penalty
    
    def _check_keywords(self, metric: str, context: str, entity_text: str) -> float:
        """FIX 5 (v3): Strict keyword validation.
        0 keyword matches → discard (0.0), 1 match → 0.7 penalty, 2+ → pass (1.0).
        """
        if metric not in METRIC_KEYWORDS:
            return 1.0
        combined = f"{entity_text} {context}".lower()
        matches = sum(1 for kw in METRIC_KEYWORDS[metric] if kw in combined)
        if matches == 0:
            return 0.0   # discard
        elif matches == 1:
            return 0.7   # weak match penalty
        return 1.0       # strong match


# ============================================================================
# POST-PROCESSING: FUZZY DEDUPLICATION (FIX 4)
# ============================================================================

class MetricDeduplicator:
    """
    FIX 4: Fuzzy deduplication.
    
    Group by (metric, unit), cluster values within ±5%, keep highest confidence.
    """
    
    @staticmethod
    def deduplicate(metrics: List[Dict]) -> List[Dict]:
        if not metrics:
            return []
        
        groups = defaultdict(list)
        for m in metrics:
            groups[(m['normalized_metric'], m.get('unit', ''))].append(m)
        
        deduped = []
        for key, group in groups.items():
            group.sort(key=lambda x: x['confidence_score'], reverse=True)
            clusters = []
            
            for entry in group:
                placed = False
                for cluster in clusters:
                    rep = cluster[0]['value']
                    if rep == 0:
                        if entry['value'] == 0:
                            cluster.append(entry)
                            placed = True
                            break
                    else:
                        if abs(entry['value'] - rep) / abs(rep) <= 0.05:
                            cluster.append(entry)
                            placed = True
                            break
                if not placed:
                    clusters.append([entry])
            
            for cluster in clusters:
                deduped.append(max(cluster, key=lambda x: x['confidence_score']))
        
        return deduped


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class ESGExtractionPipeline:
    """Complete end-to-end ESG extraction pipeline (v2)."""
    
    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None
    ):
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
        self.deduplicator = MetricDeduplicator()
    
    def process(self, pdf_path: str) -> Dict:
        """Process a PDF and extract ESG metrics."""
        results = {
            'metadata': {},
            'metrics': [],
            'discarded': [],
            'warnings': [],
            'processing_log': []
        }
        
        # Layer 1
        sections = self.pdf_extractor.extract(pdf_path)
        results['processing_log'].append(f"Extracted {len(sections)} sections")
        
        # Layer 2
        sections = self.structurer.structure(sections)
        results['metadata'] = self.structurer.extract_metadata(sections)
        
        # Layer 3
        for section in sections:
            section.text = self.preprocessor.preprocess(section.text)
        
        # Layer 4
        esg_sections = self.candidate_filter.filter_candidates(sections)
        results['processing_log'].append(
            f"Filtered to {len(esg_sections)} ESG-relevant sections"
        )
        
        # Layers 5-11
        raw_metrics = []
        for section in esg_sections:
            section_metrics = self._process_section(section)
            raw_metrics.extend(section_metrics)
        
        # Deduplication
        results['metrics'] = self.deduplicator.deduplicate(raw_metrics)
        
        # Collect warnings
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
        entities = self.ner_model.extract_entities(section.text)
        
        for entity in entities:
            # Layer 7: Context
            context = self.context_extractor.extract_context(
                section.text, entity['start'], entity['end']
            )
            
            # Layer 6: Classify
            normalized_metric, class_conf = self.classifier.classify(
                entity['text'], context,
                section_type=getattr(section, 'category', 'Unknown')
            )
            
            # Layer 8: Value extraction
            entity_pos = context.find(entity['text'])
            values = self.value_extractor.extract(
                context,
                entity_start=entity_pos if entity_pos >= 0 else None,
                entity_end=(entity_pos + len(entity['text'])) if entity_pos >= 0 else None,
            )
            
            if not values:
                continue
            
            value_data = values[0]
            
            # Layer 9: Normalize
            norm_val, norm_unit = self.unit_normalizer.normalize(
                value_data['value'], value_data['unit']
            )
            
            # Layer 11: Validate
            v_status, v_issues, v_penalty = self.validator.validate(
                metric=normalized_metric,
                value=norm_val,
                unit=norm_unit,
                context=context,
                entity_text=entity['text'],
            )
            
            # Discard if penalty is zero
            if v_penalty == 0.0:
                continue
            
            # Keyword match check
            kw_match = True
            if normalized_metric in METRIC_KEYWORDS:
                combined = f"{entity['text']} {context}".lower()
                kw_match = any(kw in combined for kw in METRIC_KEYWORDS[normalized_metric])
            
            # Layer 10: Confidence
            confidence = self.confidence_scorer.calculate(
                ner_confidence=entity['confidence'],
                classification_confidence=class_conf,
                value_confidence=value_data['confidence'],
                unit=norm_unit,
                entity_text=entity['text'],
                num_candidates=len(values),
                validation_penalty=v_penalty,
                keyword_match=kw_match,
                metric=normalized_metric,
            )
            
            metric = ExtractedMetric(
                metric_text=entity['text'],
                normalized_metric=normalized_metric,
                category="Unknown",
                value=norm_val,
                unit=norm_unit,
                context=context,
                page_number=section.page_number,
                section_id=section.section_id,
                confidence_score=confidence,
                validation_status=v_status,
                validation_messages=v_issues,
            )
            
            metrics.append(asdict(metric))
        
        return metrics


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of complete pipeline"""
    pipeline = ESGExtractionPipeline(
        ner_model_path='./models/ner_model',
        classifier_model_path='./models/classifier_model'
    )
    results = pipeline.process('example_esg_report.pdf')
    
    with open('extracted_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Extracted {len(results['metrics'])} metrics")
    print(f"Warnings: {len(results['warnings'])}")


if __name__ == "__main__":
    main()
