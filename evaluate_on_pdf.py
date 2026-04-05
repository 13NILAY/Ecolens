"""
ESG EXTRACTION - PDF EVALUATION SCRIPT (v2 — Rule-Based Quality Improvements)
================================================================================

Improvements over v1:
  1. Context-aware value extraction (entity proximity, sentence-based)
  2. Strict unit validation (whitelist + metric-unit compatibility)
  3. Invalid numeric pattern filtering (years, page numbers, broken notation)
  3b. Multiple-value handling with temporal keyword awareness
  4. Fuzzy deduplication (±5% value clustering)
  5. Sentence-based context window extraction
  6. Improved classifier input format
  7. Hard validation rules (domain constraints, keyword validation, unitless rules)
  8. Confidence scoring overhaul (penalties + [0.3, 0.95] clamp)
  9. Balanced ESG candidate filtering

Usage:
    python evaluate_on_pdf.py --pdf_path "path/to/esg_report.pdf"
    python evaluate_on_pdf.py --pdf_dir "path/to/pdfs/"

Requirements:
    pip install pdfplumber transformers torch
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import re
from dataclasses import dataclass, asdict
from collections import defaultdict

# PDF extraction
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: pdfplumber not installed. Install with: pip install pdfplumber")
    PDF_AVAILABLE = False

# Model inference
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)
import numpy as np


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# 🎯 TARGET METRICS - Only extract these 7
TARGET_METRICS: Set[str] = {
    'SCOPE_1',
    'SCOPE_2', 
    'SCOPE_3',
    'ESG_SCORE',
    'WASTE_GENERATED',
    'ENERGY_CONSUMPTION',
    'WATER_USAGE'
}


# --- FIX 2: Strict Unit Whitelist ---
VALID_UNITS: Set[str] = {
    "tCO2e", "Mt CO2e",
    "%",
    "GWh", "MWh", "kWh", "Mn kWh",
    "MJ", "GJ", "TJ",
    "m³", "m3", "KL", "kilolitres", "liters", "litres",
    "kg", "tonnes", "t", "MT",
    "crore", "lakh",
}

# --- FIX 2: Unit normalization map ---
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

# --- Metric-Unit Compatibility Map ---
METRIC_UNIT_MAP: Dict[str, Set[str]] = {
    'SCOPE_1': {'tCO2e', 'Mt CO2e', 'tonnes', 't', 'MT', 'kg'},
    'SCOPE_2': {'tCO2e', 'Mt CO2e', 'tonnes', 't', 'MT', 'kg'},
    'SCOPE_3': {'tCO2e', 'Mt CO2e', 'tonnes', 't', 'MT', 'kg'},
    'ENERGY_CONSUMPTION': {'GWh', 'MWh', 'kWh', 'MJ', 'GJ', 'TJ', 'Mn kWh', 'crore'},
    'WATER_USAGE': {'m³', 'KL', 'kilolitres', 'liters', 'litres', 'crore'},
    'WASTE_GENERATED': {'tonnes', 'MT', 't', 'kg'},
    'ESG_SCORE': set(),  # unitless allowed
}

# --- Metric-Specific Keywords ---
METRIC_KEYWORDS: Dict[str, List[str]] = {
    'SCOPE_1': ['emissions', 'co2', 'ghg', 'carbon', 'scope 1', 'scope1', 'direct emissions'],
    'SCOPE_2': ['emissions', 'co2', 'ghg', 'carbon', 'scope 2', 'scope2', 'indirect emissions'],
    'SCOPE_3': ['emissions', 'co2', 'ghg', 'carbon', 'scope 3', 'scope3', 'value chain'],
    'ENERGY_CONSUMPTION': ['energy', 'electricity', 'power', 'consumption', 'kwh', 'mwh', 'gwh'],
    'WATER_USAGE': ['water', 'withdrawal', 'discharge', 'consumption'],
    'WASTE_GENERATED': ['waste', 'generated', 'disposal', 'solid waste'],
    'ESG_SCORE': ['esg', 'score', 'rating', 'sustainability', 'index'],
}

# --- Metrics allowed without units ---
UNITLESS_ALLOWED_METRICS: Set[str] = {
    'ESG_SCORE',
}

# --- FIX 3: Temporal/comparison keywords ---
TEMPORAL_KEYWORDS: List[str] = [
    'compared', 'previous', 'last year', 'prior year', 'preceding',
    'increase', 'decrease', 'growth', 'decline', 'change',
    'fy 2022', 'fy 2021', 'fy 2020',  # previous FY markers
]

# --- FIX 3: Words that invalidate a unit ---
INVALID_UNIT_WORDS: Set[str] = {
    'the', 'a', 'an', 'and', 'or', 'for', 'of', 'in', 'to', 'by', 'on', 'at',
    'is', 'it', 'as', 'be', 'was', 'has', 'had', 'are', 'its', 'our', 'we',
    'all', 'key', 'no', 'yes', 'na', 'nil',
    'total', 'from', 'note', 'business', 'gri', 'fy', 'dry', 'bio',
    'male', 'female',  # contextual words, not units
    'accounts', 'carbon', 'emissions', 'turnover',  # metric words misread as units
    'worked', 'compared', 'standards', 'awareness',
    'e', 'p', 'g', 'r', 's',  # single letters
}

# --- FIX 8: Generic entity words that reduce confidence ---
GENERIC_ENTITY_WORDS: Set[str] = {
    'total', 'number', 'energy', 'water', 'waste', 'carbon', 'scope',
    'paid', 'value', 'women', 'employee', 'os', 'km',
}


# ============================================================================
# LAYER 1: PDF TEXT EXTRACTION
# ============================================================================

class PDFExtractor:
    """Extract text from PDF documents"""
    
    def __init__(self):
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
    
    def extract_text(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF
        
        Returns:
        {
            'full_text': str,
            'pages': [{'page_num': int, 'text': str}],
            'metadata': {...}
        }
        """
        result = {
            'full_text': '',
            'pages': [],
            'metadata': {}
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            result['metadata'] = {
                'num_pages': len(pdf.pages),
                'file_path': str(pdf_path)
            }
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                
                if page_text:
                    result['pages'].append({
                        'page_num': i + 1,
                        'text': page_text
                    })
                    result['full_text'] += page_text + '\n\n'
        
        print(f"✓ Extracted text from {len(result['pages'])} pages")
        return result


# ============================================================================
# LAYER 2-3: PREPROCESSING & STRUCTURING
# ============================================================================

class TextPreprocessor:
    """Clean and structure extracted text"""
    
    def __init__(self):
        self.esg_section_keywords = {
            'Environmental': ['environmental', 'climate', 'emissions', 'carbon', 'energy', 'water', 'waste'],
            'Social': ['social', 'employees', 'diversity', 'safety', 'training', 'labor'],
            'Governance': ['governance', 'board', 'ethics', 'compliance', 'risk']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean PDF artifacts"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix broken words
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        # Remove page numbers
        text = re.sub(r'\n\d+\n', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[Dict]:
        """Chunk text into overlapping segments for processing"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to break at sentence boundary
            if end < len(text):
                period_pos = text.rfind('.', start, end)
                if period_pos > start + max_length - 100:
                    end = period_pos + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                section_type = self._classify_section(chunk_text)
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'section_type': section_type
                })
            
            start = end - overlap
        
        return chunks
    
    def _classify_section(self, text: str) -> str:
        """Classify text section as Environmental, Social, or Governance"""
        text_lower = text.lower()
        
        scores = {
            'Environmental': sum(text_lower.count(kw) for kw in self.esg_section_keywords['Environmental']),
            'Social': sum(text_lower.count(kw) for kw in self.esg_section_keywords['Social']),
            'Governance': sum(text_lower.count(kw) for kw in self.esg_section_keywords['Governance'])
        }
        
        if max(scores.values()) == 0:
            return 'Unknown'
        
        return max(scores, key=scores.get)


# ============================================================================
# LAYER 4: ESG CANDIDATE FILTER (FIX 9 — Balanced)
# ============================================================================

class ESGCandidateFilter:
    """
    Filter chunks to only ESG-relevant content.
    
    FIX 9: Balanced filtering — 2-keyword threshold, no numeric requirement,
    numeric presence boosts confidence.
    """
    
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        self.threshold = threshold
        
        if model_path and Path(model_path).exists():
            print(f"Loading ESG filter model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            self.use_model = True
        else:
            print("⚠️  ESG filter model not found. Using keyword-based filter.")
            self.use_model = False
            self._init_keyword_filter()
    
    def _init_keyword_filter(self):
        """Fallback keyword-based filter"""
        self.esg_keywords = {
            'emissions', 'carbon', 'co2', 'ghg', 'scope', 'energy', 'renewable',
            'water', 'waste', 'recycling', 'employees', 'workforce', 'diversity',
            'safety', 'training', 'board', 'governance', 'ethics', 'compliance', 'esg',
            'sustainability', 'injury', 'turnover', 'attrition',
        }
        # Negative keywords — financial/legal terms that generate false positives
        self.negative_keywords = {
            'revenue', 'profit', 'earnings', 'ebitda', 'cash flow', 'dividend',
            'stock price', 'market cap', 'valuation', 'debt',
            'litigation', 'lawsuit', 'arbitration',
        }
    
    def is_esg_candidate(self, text: str) -> Tuple[bool, float]:
        """Check if text is ESG-related"""
        if self.use_model:
            return self._model_filter(text)
        else:
            return self._keyword_filter(text)
    
    def _model_filter(self, text: str) -> Tuple[bool, float]:
        """Model-based filtering"""
        inputs = self.tokenizer(
            text, max_length=256, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            esg_prob = probs[0][1].item()
        return esg_prob > self.threshold, esg_prob
    
    def _keyword_filter(self, text: str) -> Tuple[bool, float]:
        """
        Keyword-based filtering.
        
        FIX 9:
        - Minimum 2 ESG keywords (NOT 3)
        - No numeric requirement for candidacy
        - Numeric presence boosts confidence
        """
        text_lower = text.lower()
        keyword_count = sum(1 for kw in self.esg_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        # Must have at least 2 ESG keywords
        is_esg = keyword_count >= 2
        
        # Negative keywords reduce confidence but don't disqualify
        base_confidence = min(keyword_count / 5.0, 1.0)
        
        # Apply confidence threshold
        if base_confidence < 0.4:
            is_esg = False
        if negative_count > 0:
            base_confidence *= max(0.5, 1.0 - negative_count * 0.15)
        
        # FIX 9: Numeric presence boosts confidence (but is NOT required)
        has_number = bool(re.search(r'\d+\.?\d*', text))
        if has_number:
            base_confidence = min(1.0, base_confidence + 0.2)
        
        return is_esg, base_confidence


# ============================================================================
# LAYER 5: NER MODEL
# ============================================================================

class ESGNERExtractor:
    """Extract ESG metric entities using trained NER model"""
    
    def __init__(self, model_path: str):
        print(f"Loading NER model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        label_map_path = Path(model_path).parent / 'label_mappings.json'
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                mappings = json.load(f)
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        else:
            self.id2label = self.model.config.id2label
        
        print(f"✓ NER model loaded with {len(self.id2label)} labels")
    
    def extract_entities(self, text: str, section_type: str = "Unknown") -> List[Dict]:
        """Extract metric entities from text"""
        inputs = self.tokenizer(
            text, max_length=512, padding=True, truncation=True,
            return_tensors="pt", return_offsets_mapping=True
        )
        offset_mapping = inputs.pop('offset_mapping')[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
            probs = torch.softmax(outputs.logits, dim=2)[0]
        
        entities = []
        current_entity = None
        
        for idx, (pred_id, prob_dist) in enumerate(zip(predictions, probs)):
            label = self.id2label.get(pred_id.item(), 'O')
            confidence = prob_dist[pred_id].item()
            
            if offset_mapping[idx][0] == offset_mapping[idx][1]:
                continue
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_label = label[2:]
                start, end = offset_mapping[idx]
                current_entity = {
                    'text': text[start:end],
                    'label': entity_label,
                    'start': start.item(),
                    'end': end.item(),
                    'confidence': confidence,
                    'section_type': section_type
                }
            elif label.startswith('I-') and current_entity:
                entity_label = label[2:]
                if entity_label == current_entity['label']:
                    _, end = offset_mapping[idx]
                    current_entity['end'] = end.item()
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                    current_entity['confidence'] = min(current_entity['confidence'], confidence)
            elif label == 'O' and current_entity:
                entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities


# ============================================================================
# LAYER 6: METRIC CLASSIFIER (FIX 6 — Improved Input Format)
# ============================================================================

class MetricClassifier:
    """Classify extracted metrics to normalized labels"""
    
    def __init__(self, model_path: str):
        print(f"Loading classifier model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        label_map_path = Path(model_path).parent / 'label_mappings.json'
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                mappings = json.load(f)
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        else:
            self.id2label = self.model.config.id2label
        
        print(f"✓ Classifier loaded with {len(self.id2label)} classes")
    
    def classify(self, metric_text: str, context: str, section_type: str = "") -> Tuple[str, float]:
        """
        Classify metric to normalized label.
        
        FIX 6: Improved input format for better disambiguation.
        Old: "[Environmental] Scope 2 emissions [SEP] <raw context>"
        New: "[SECTION: Environmental] METRIC: Scope 2 emissions CONTEXT: <sentence_context>"
        """
        input_text = (
            f"[SECTION: {section_type}] "
            f"METRIC: {metric_text} "
            f"CONTEXT: {context}"
        )
        
        inputs = self.tokenizer(
            input_text, max_length=128, padding=True, truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
        
        normalized_metric = self.id2label.get(pred_id, 'UNKNOWN')
        # 🎯 V3 FIX 1: Only return target metrics
        if normalized_metric not in TARGET_METRICS:
            return "UNKNOWN", 0.0

        return normalized_metric, confidence


# ============================================================================
# LAYER 7-8: CONTEXT + VALUE EXTRACTION (FIXES 1, 2, 3, 3b, 5)
# ============================================================================

class ValueExtractor:
    """
    Extract values and units from context using rule-based approach.
    
    Improvements:
    - FIX 1: Entity-proximity extraction (±50 chars, closest value)
    - FIX 2: Strict unit whitelist validation
    - FIX 3: Invalid numeric pattern rejection
    - FIX 3b: Temporal keyword awareness for multi-value disambiguation
    - FIX 5: Sentence-based context windows
    """
    
    def __init__(self):
        # Regex patterns ordered by specificity
        self.patterns = [
            # Number with unit (e.g., "1,200.5 tCO2e", "74%")
            (r'([\d,]+\.?\d*)\s*(%)', 'pct'),
            (r'([\d,]+\.\d+)\s+([a-zA-Z/%³]+(?:\s+[a-zA-Z]+)?)', 'full_float'),
            (r'([\d,]+)\s+([a-zA-Z/%³]+(?:\s+[a-zA-Z]+)?)', 'full_int'),
            # Standalone number
            (r'([\d,]+\.\d+)', 'float_only'),
            (r'([\d,]+)', 'int_only'),
        ]
        
        self.confidence_map = {
            'pct': 0.95,
            'full_float': 0.90,
            'full_int': 0.85,
            'float_only': 0.60,
            'int_only': 0.50,
        }
    
    # ---- FIX 5: Sentence-based context extraction ----
    
    def extract_context(self, text: str, entity_start: int, entity_end: int, window: int = 150) -> str:
        """
        FIX 5: Extract context as full sentence(s) containing the entity.
        Falls back to ±window character extraction if sentence detection fails.
        """
        # Try sentence-based extraction
        sentence_context = self._extract_sentence_context(text, entity_start, entity_end)
        if sentence_context and len(sentence_context) >= 30:
            return sentence_context
        
        # Fallback to character window
        context_start = max(0, entity_start - window)
        context_end = min(len(text), entity_end + window)
        return text[context_start:context_end]
    
    def _extract_sentence_context(self, text: str, entity_start: int, entity_end: int) -> Optional[str]:
        """Split text into sentences and return sentence(s) containing the entity + next sentence."""
        # Split by sentence boundaries
        sentence_boundaries = list(re.finditer(r'[.!?]\s+', text))
        
        if not sentence_boundaries:
            return None
        
        # Build sentence spans
        sentences = []
        prev_end = 0
        for match in sentence_boundaries:
            sentences.append((prev_end, match.end()))
            prev_end = match.end()
        # Last sentence
        if prev_end < len(text):
            sentences.append((prev_end, len(text)))
        
        # Find sentence containing entity
        entity_sentence_idx = None
        for i, (s_start, s_end) in enumerate(sentences):
            if s_start <= entity_start < s_end:
                entity_sentence_idx = i
                break
        
        if entity_sentence_idx is None:
            return None
        
        # Return entity sentence + next sentence (if exists)
        start = sentences[entity_sentence_idx][0]
        end_idx = min(entity_sentence_idx + 1, len(sentences) - 1)
        end = sentences[end_idx][1]
        
        return text[start:end].strip()
    
    # ---- FIX 1: Entity-proximity value extraction ----
    
    def extract_value(
        self,
        context: str,
        entity_start_in_context: Optional[int] = None,
        entity_end_in_context: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        FIX 1: Context-aware value extraction.
        
        - Only consider values within ±50 chars of entity span
        - Rank by proximity to entity
        - FIX 3: Reject invalid patterns (years, page numbers, broken notation)
        - FIX 3b: Temporal keyword awareness
        
        Returns: {'value': float, 'unit': str, 'confidence': float, 'num_candidates': int}
        """
        raw_candidates = self._extract_all_candidates(context)
        
        if not raw_candidates:
            return None
        
        # FIX 3: Filter invalid patterns
        candidates = [c for c in raw_candidates if self._is_valid_extraction(c, context)]
        
        if not candidates:
            return None
        
        # FIX 1: Proximity filtering (if entity position is known)
        if entity_start_in_context is not None and entity_end_in_context is not None:
            entity_center = (entity_start_in_context + entity_end_in_context) / 2
            
            # Score by distance to entity
            for c in candidates:
                match_center = (c['match_start'] + c['match_end']) / 2
                distance = abs(match_center - entity_center)
                c['distance'] = distance
                
                # Proximity bonus: within ±50 chars is strongly preferred
                if distance <= 50:
                    c['proximity_score'] = 1.0
                elif distance <= 100:
                    c['proximity_score'] = 0.7
                else:
                    c['proximity_score'] = 0.4
            
            # FIX 3b: Penalize values appearing after temporal keywords
            for c in candidates:
                text_before_value = context[max(0, c['match_start'] - 60):c['match_start']].lower()
                if any(kw in text_before_value for kw in TEMPORAL_KEYWORDS):
                    c['proximity_score'] *= 0.5  # Likely a comparison/previous year value
            
            # Sort by proximity (closest first), then by extraction confidence
            candidates.sort(key=lambda c: (-c['proximity_score'], c['distance']))
        
        best = candidates[0]
        num_candidates = len(candidates)
        
        result = {
            'value': best['value'],
            'unit': best['unit'],
            'confidence': best['confidence'],
            'num_candidates': num_candidates,
        }
        
        # FIX 3b: Penalize if multiple candidates exist
        if num_candidates > 1:
            result['confidence'] *= 0.6
        
        return result
    
    def _extract_all_candidates(self, context: str) -> List[Dict]:
        """Extract all (value, unit) candidate pairs from context."""
        candidates = []
        seen_spans = set()  # Avoid duplicate matches from overlapping patterns
        
        for pattern, pattern_type in self.patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            
            for match in matches:
                span = (match.start(), match.end())
                
                # Skip if this span overlaps with an already-captured higher-priority match
                if any(self._spans_overlap(span, s) for s in seen_spans):
                    continue
                
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                except (ValueError, IndexError):
                    continue
                
                unit = ""
                if len(match.groups()) >= 2 and match.group(2):
                    unit_raw = match.group(2).strip()
                    unit = self._normalize_unit(unit_raw)
                
                # FIX 2: Validate unit against whitelist
                if unit and not self._is_valid_unit(unit):
                    continue  # Discard — completely remove invalid units
                
                candidates.append({
                    'value': value,
                    'unit': unit,
                    'confidence': self.confidence_map[pattern_type],
                    'match_start': match.start(),
                    'match_end': match.end(),
                })
                seen_spans.add(span)
        
        return candidates
    
    def _normalize_unit(self, unit_raw: str) -> str:
        """Normalize unit string to canonical form."""
        unit_lower = unit_raw.lower().strip()
        return UNIT_NORMALIZATION.get(unit_lower, unit_raw)
    
    def _is_valid_unit(self, unit: str) -> bool:
        """
        FIX 2: Check if unit is in the whitelist.
        Also rejects single-letter units and common English words.
        """
        # Check against invalid words
        if unit.lower() in INVALID_UNIT_WORDS:
            return False
        
        # Single character units are almost always garbage
        if len(unit) <= 1 and unit != '%':
            return False
        
        # Check against whitelist
        if unit in VALID_UNITS:
            return True
        
        # Check normalized form
        normalized = UNIT_NORMALIZATION.get(unit.lower(), '')
        if normalized in VALID_UNITS:
            return True
        
        return False
    
    def _is_valid_extraction(self, candidate: Dict, context: str) -> bool:
        """
        FIX 3: Reject invalid numeric patterns.
        
        Rejects:
        - Years (1900-2030)
        - Page numbers (when "page" is nearby)
        - Broken scientific notation ("46.7 E")
        - Reference numbers (after "GRI", "section", "principle")
        """
        value = candidate['value']
        match_start = candidate['match_start']
        
        # FIX 5 (page/year noise): Reject years
        if 1900 <= value <= 2030 and value == int(value):
            return False
        
        # Reject likely page numbers
        text_nearby = context[max(0, match_start - 30):match_start].lower()
        if value < 300 and value == int(value):
            if any(kw in text_nearby for kw in ['page', 'p.', 'section', 'principle', 'gri ']):
                return False
        
        # Reject reference numbers (GRI 305-1, ISO 14001, etc.)
        if re.search(r'(?:gri|iso|p\d|e\d)\s*$', text_nearby, re.IGNORECASE):
            return False
        
        # Reject FY markers (FY 2023-24 → "24" extracted)
        if re.search(r'fy\s*$', text_nearby, re.IGNORECASE) and value < 100:
            return False
        
        return True
    
    @staticmethod
    def _spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two character spans overlap."""
        return span1[0] < span2[1] and span2[0] < span1[1]


# ============================================================================
# LAYER 11: VALIDATION (FIX 7 — Hard Rules + Metric-Unit + Keywords)
# ============================================================================

class MetricValidator:
    """
    Validate extracted metrics with strict domain rules.
    
    FIX 7: Hard validation rules
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
        Comprehensive validation.
        
        Returns: (status, issues, confidence_penalty_multiplier)
        
        status: "VALID", "WARNING", "INVALID"
        confidence_penalty_multiplier: multiply into confidence score
        """
        issues = []
        status = "VALID"
        penalty = 1.0
        
        # ---- FIX 7: Range validation ----
        if metric in self.value_ranges:
            min_val, max_val = self.value_ranges[metric]
            
            if value < min_val:
                issues.append(f"Value {value} below minimum {min_val}")
                status = "INVALID"
                penalty *= 0.5
            
            if value > max_val:
                issues.append(f"Value {value} above maximum {max_val}")
                status = "INVALID"
                penalty *= 0.5
        
        # ESG Score range check
        if metric == 'ESG_SCORE' and not (0 <= value <= 100):
            issues.append(f"ESG Score {value} out of range [0, 100]")
            status = "INVALID"
            penalty *= 0.3
        
        # ---- FIX 7c: Metric-unit compatibility ----
        if metric in METRIC_UNIT_MAP:
            valid_units_for_metric = METRIC_UNIT_MAP[metric]
            
            if valid_units_for_metric:  # Non-empty set means we enforce
                if unit and unit not in valid_units_for_metric:
                    issues.append(f"Unit '{unit}' incompatible with metric '{metric}'")
                    status = "INVALID"
                    penalty *= 0.0  # Complete discard
        
        # ---- FIX 7e: Unitless metric rules ----
        if not unit:
            if metric not in UNITLESS_ALLOWED_METRICS:
                issues.append(f"Missing unit for metric '{metric}' (not a score metric)")
                status = "INVALID"
                penalty *= 0.0  # Complete discard
        
        # ---- FIX 7d: Metric-specific keyword validation ----
        keyword_penalty = self._check_metric_keywords(metric, context, entity_text)
        if keyword_penalty < 1.0:
            if keyword_penalty == 0.0:
                issues.append(f"No domain keywords found for metric '{metric}'")
                status = "INVALID" if status != "INVALID" else status
            else:
                issues.append(f"Weak keyword match for metric '{metric}'")
                if status == "VALID":
                    status = "WARNING"
            penalty *= keyword_penalty
        
        return status, issues, penalty
    
    def _check_metric_keywords(self, metric: str, context: str, entity_text: str) -> float:
        """
        FIX 7d: Check if context/entity contains domain-relevant keywords.
        
        Returns: 1.0 (full match), 0.5 (weak), 0.0 (no match → discard)
        """
        if metric not in METRIC_KEYWORDS:
            return 1.0  # No keywords defined for this metric
        
        required_keywords = METRIC_KEYWORDS[metric]
        combined_text = f"{entity_text} {context}".lower()
        
        matches = sum(1 for kw in required_keywords if kw in combined_text)
        
        if matches >= 2:
            return 1.0  # Strong match
        elif matches == 1:
            return 0.7  # Weak but acceptable
        else:
            return 0.0  # No domain keywords at all — discard


# ============================================================================
# LAYER 10: CONFIDENCE SCORING (FIX 8 — Overhaul)
# ============================================================================

class ConfidenceScorer:
    """
    Aggregate confidence with realistic penalties.
    
    FIX 8: Weighted average + cumulative penalties + hard clamp [0.3, 0.95].
    """
    
    def calculate(
        self,
        ner_confidence: float,
        classification_confidence: float,
        value_confidence: float,
        unit: str,
        entity_text: str,
        num_value_candidates: int,
        validation_penalty: float,
        keyword_match: bool = True,
    ) -> float:
        """
        Calculate aggregate confidence with penalties.
        """
        # Base weighted average
        base_score = (
            ner_confidence * 0.4 +
            classification_confidence * 0.3 +
            value_confidence * 0.3
        )
        
        # ---- Apply penalties ----
        
        # Missing unit
        if not unit:
            base_score *= 0.7
        
        # Validation failure
        base_score *= validation_penalty
        
        # Multiple value candidates in context
        if num_value_candidates > 1:
            base_score *= 0.6
        
        # Short entity text (likely garbage NER output like "os", "KM")
        if len(entity_text.strip()) < 3:
            base_score *= 0.6
        
        # Generic entity word
        if entity_text.strip().lower() in GENERIC_ENTITY_WORDS:
            base_score *= 0.7
        
        # Weak keyword match
        if not keyword_match:
            base_score *= 0.5
        
        # ---- FIX 8: Hard clamp ----
        confidence = max(0.3, min(0.95, base_score))
        
        return round(confidence, 4)


# ============================================================================
# LAYER POST: DEDUPLICATION (FIX 4 — Fuzzy ±5%)
# ============================================================================

class MetricDeduplicator:
    """
    FIX 4: Fuzzy deduplication.
    
    Group by (normalized_metric, unit), cluster values within ±5%,
    keep highest confidence entry per cluster.
    """
    
    @staticmethod
    def deduplicate(metrics: List[Dict]) -> List[Dict]:
        """Remove duplicates using fuzzy value matching."""
        if not metrics:
            return []
        
        # Group by (metric, unit)
        groups = defaultdict(list)
        for m in metrics:
            key = (m['normalized_metric'], m.get('unit', ''))
            groups[key].append(m)
        
        deduped = []
        
        for (metric, unit), group in groups.items():
            # Sort by confidence descending
            group.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Cluster values within ±5%
            clusters = []
            for entry in group:
                placed = False
                for cluster in clusters:
                    representative_value = cluster[0]['value']
                    if representative_value == 0:
                        if entry['value'] == 0:
                            cluster.append(entry)
                            placed = True
                            break
                    else:
                        pct_diff = abs(entry['value'] - representative_value) / abs(representative_value)
                        if pct_diff <= 0.05:  # Within ±5%
                            cluster.append(entry)
                            placed = True
                            break
                
                if not placed:
                    clusters.append([entry])
            
            # Keep highest confidence from each cluster
            for cluster in clusters:
                best = max(cluster, key=lambda x: x['confidence'])
                deduped.append(best)
        
        return deduped


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class PDFEvaluationPipeline:
    """Complete pipeline for evaluating PDFs (v2 with all quality fixes)"""
    
    def __init__(
        self,
        esg_filter_path: Optional[str] = None,
        ner_model_path: str = './models/ner_model/final',
        classifier_path: str = './models/classifier/final'
    ):
        print("\n" + "="*70)
        print("INITIALIZING ESG EXTRACTION PIPELINE v2")
        print("="*70)
        
        self.pdf_extractor = PDFExtractor()
        self.preprocessor = TextPreprocessor()
        self.esg_filter = ESGCandidateFilter(esg_filter_path)
        self.ner_extractor = ESGNERExtractor(ner_model_path)
        self.classifier = MetricClassifier(classifier_path)
        self.value_extractor = ValueExtractor()
        self.validator = MetricValidator()
        self.confidence_scorer = ConfidenceScorer()
        self.deduplicator = MetricDeduplicator()
        
        print("✓ Pipeline v2 initialized (with rule-based quality fixes)")
    
    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a PDF and extract ESG metrics"""
        print("\n" + "="*70)
        print(f"PROCESSING: {pdf_path}")
        print("="*70)
        
        results = {
            'file': str(pdf_path),
            'pipeline_version': 'v2_rule_based_fixes',
            'metrics': [],
            'discarded': [],
            'statistics': {},
            'warnings': []
        }
        
        # Layer 1: Extract text
        print("\n[Layer 1] Extracting text from PDF...")
        pdf_data = self.pdf_extractor.extract_text(pdf_path)
        
        # Layer 2-3: Preprocess and chunk
        print("[Layer 2-3] Preprocessing and chunking text...")
        clean_text = self.preprocessor.clean_text(pdf_data['full_text'])
        chunks = self.preprocessor.chunk_text(clean_text)
        print(f"✓ Created {len(chunks)} text chunks")
        
        # Layer 4: Filter ESG candidates
        print("[Layer 4] Filtering ESG candidates...")
        esg_chunks = []
        for chunk in chunks:
            is_esg, confidence = self.esg_filter.is_esg_candidate(chunk['text'])
            if is_esg:
                chunk['esg_confidence'] = confidence
                esg_chunks.append(chunk)
        
        esg_pct = len(esg_chunks) / len(chunks) * 100 if chunks else 0
        print(f"✓ {len(esg_chunks)} / {len(chunks)} chunks are ESG-related ({esg_pct:.1f}%)")
        
        # Process each ESG chunk
        all_metrics = []
        all_discarded = []
        
        for chunk_idx, chunk in enumerate(esg_chunks):
            # Layer 5: NER extraction
            entities = self.ner_extractor.extract_entities(
                chunk['text'], chunk['section_type']
            )
            
            if not entities:
                continue
            
            for entity in entities:
                metric_result, discard_reason = self._process_entity(entity, chunk)
                
                if discard_reason:
                    all_discarded.append({
                        'entity_text': entity['text'],
                        'reason': discard_reason,
                        'chunk_idx': chunk_idx,
                    })
                elif metric_result:
                    all_metrics.append(metric_result)
        
        print(f"\n[Pre-dedup] Raw extractions: {len(all_metrics)}, Discarded: {len(all_discarded)}")
        
        # FIX 4: Fuzzy deduplication
        results['metrics'] = self.deduplicator.deduplicate(all_metrics)
        results['metrics'].sort(key=lambda x: x['confidence'], reverse=True)
        results['discarded'] = all_discarded
        
        print(f"[Post-dedup] Unique metrics: {len(results['metrics'])}")
        
        # Calculate statistics
        valid_metrics = [m for m in results['metrics'] if m['validation_status'] == 'VALID']
        confidences = [m['confidence'] for m in results['metrics']]
        
        results['statistics'] = {
            'total_chunks': len(chunks),
            'esg_chunks': len(esg_chunks),
            'raw_extractions': len(all_metrics),
            'discarded_extractions': len(all_discarded),
            'final_metrics': len(results['metrics']),
            'valid_metrics': len(valid_metrics),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0,
            'min_confidence': float(np.min(confidences)) if confidences else 0,
            'max_confidence': float(np.max(confidences)) if confidences else 0,
        }
        
        # Collect warnings
        for metric in results['metrics']:
            if metric['validation_status'] != 'VALID':
                results['warnings'].append({
                    'metric': metric['normalized_metric'],
                    'issues': metric['validation_issues']
                })
        
        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {output_path}")
        
        self._print_summary(results)
        return results
    
    def _process_entity(self, entity: Dict, chunk: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Process a single entity through layers 6-11 with all quality gates.
        
        Returns: (metric_result_or_None, discard_reason_or_None)
        """
        # ---- Layer 7: Extract context (FIX 5: sentence-based) ----
        context = self.value_extractor.extract_context(
            chunk['text'], entity['start'], entity['end']
        )
        
        # ---- Layer 6: Classify metric (FIX 6: improved input) ----
        normalized_metric, class_conf = self.classifier.classify(
            entity['text'], context, entity['section_type']
        )
        
        # ---- Layer 7-8: Extract value (FIXES 1, 2, 3, 3b) ----
        
        # Calculate entity position within context for proximity scoring
        entity_text_in_context = entity['text']
        entity_pos_in_context = context.find(entity_text_in_context)
        
        if entity_pos_in_context >= 0:
            ent_start_ctx = entity_pos_in_context
            ent_end_ctx = entity_pos_in_context + len(entity_text_in_context)
        else:
            ent_start_ctx = None
            ent_end_ctx = None
        
        value_data = self.value_extractor.extract_value(
            context,
            entity_start_in_context=ent_start_ctx,
            entity_end_in_context=ent_end_ctx,
        )
        
        if not value_data:
            return None, "no_valid_value_found"
        
        # ---- Layer 11: Validate (FIX 7: hard rules) ----
        validation_status, validation_issues, validation_penalty = self.validator.validate(
            metric=normalized_metric,
            value=value_data['value'],
            unit=value_data['unit'],
            context=context,
            entity_text=entity['text'],
        )
        
        # Complete discard if penalty is 0
        if validation_penalty == 0.0:
            return None, f"validation_discard: {'; '.join(validation_issues)}"
        
        # Check if keyword match is present (for confidence scoring)
        keyword_match = True
        if normalized_metric in METRIC_KEYWORDS:
            combined = f"{entity['text']} {context}".lower()
            keyword_match = any(kw in combined for kw in METRIC_KEYWORDS[normalized_metric])
        
        # ---- Layer 10: Confidence scoring (FIX 8) ----
        confidence = self.confidence_scorer.calculate(
            ner_confidence=entity['confidence'],
            classification_confidence=class_conf,
            value_confidence=value_data['confidence'],
            unit=value_data['unit'],
            entity_text=entity['text'],
            num_value_candidates=value_data.get('num_candidates', 1),
            validation_penalty=validation_penalty,
            keyword_match=keyword_match,
        )
        
        # Build result
        metric_result = {
            'metric_text': entity['text'],
            'normalized_metric': normalized_metric,
            'value': value_data['value'],
            'unit': value_data['unit'],
            'confidence': confidence,
            'section_type': entity['section_type'],
            'validation_status': validation_status,
            'validation_issues': validation_issues,
            'context': context[:200] + '...' if len(context) > 200 else context,
        }
        
        return metric_result, None
    
    def _print_summary(self, results: Dict):
        """Print extraction summary"""
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY (v2)")
        print("="*70)
        
        stats = results['statistics']
        print(f"\nText Processing:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  ESG chunks: {stats['esg_chunks']} ({stats['esg_chunks']/max(stats['total_chunks'],1)*100:.1f}%)")
        
        print(f"\nExtraction Quality:")
        print(f"  Raw extractions: {stats['raw_extractions']}")
        print(f"  Discarded (quality gates): {stats['discarded_extractions']}")
        print(f"  Final unique metrics: {stats['final_metrics']}")
        print(f"  Valid metrics: {stats['valid_metrics']}")
        
        print(f"\nConfidence Distribution:")
        print(f"  Avg: {stats['avg_confidence']:.2%}")
        print(f"  Min: {stats['min_confidence']:.2%}")
        print(f"  Max: {stats['max_confidence']:.2%}")
        
        if results['metrics']:
            print(f"\nTop Extractions:")
            for i, metric in enumerate(results['metrics'][:10], 1):
                status_icon = "✓" if metric['validation_status'] == 'VALID' else "⚠"
                unit_display = metric['unit'] if metric['unit'] else "(no unit)"
                print(f"  {status_icon} {metric['normalized_metric']}: "
                      f"{metric['value']} {unit_display} "
                      f"(conf: {metric['confidence']:.2%}, {metric['validation_status']})")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings: {len(results['warnings'])}")
            for w in results['warnings'][:5]:
                print(f"  - {w['metric']}: {', '.join(w['issues'])}")
        
        if results['discarded']:
            print(f"\n🗑️  Discarded: {len(results['discarded'])} extractions")
            # Show discard reason distribution
            reasons = defaultdict(int)
            for d in results['discarded']:
                reason_key = d['reason'].split(':')[0]
                reasons[reason_key] += 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate ESG extraction on PDF documents (v2)')
    parser.add_argument('--pdf_path', type=str, help='Path to single PDF file')
    parser.add_argument('--pdf_dir', type=str, help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--esg_filter', type=str, default='./models/esg_filter/final', help='ESG filter model path')
    parser.add_argument('--ner_model', type=str, default='./models/ner_model/final', help='NER model path')
    parser.add_argument('--classifier', type=str, default='./models/classifier/final', help='Classifier model path')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PDFEvaluationPipeline(
        esg_filter_path=args.esg_filter if Path(args.esg_filter).exists() else None,
        ner_model_path=args.ner_model,
        classifier_path=args.classifier
    )
    
    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        output_path = output_dir / f"{pdf_path.stem}_results_v2.json"
        pipeline.process_pdf(str(pdf_path), str(output_path))
    
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob('*.pdf'))
        
        print(f"\nFound {len(pdf_files)} PDF files")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n{'='*70}")
            print(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            print(f"{'='*70}")
            
            output_path = output_dir / f"{pdf_path.stem}_results_v2.json"
            pipeline.process_pdf(str(pdf_path), str(output_path))
    
    else:
        print("Error: Provide either --pdf_path or --pdf_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
