"""
ESG EXTRACTION - PDF EVALUATION SCRIPT v8.0 (TOTAL-ONLY ESG INTELLIGENCE)
================================================================================

Based on v7.1 with critical improvements for total-value extraction:
  ✅ FIX 33: TABLE OVERRIDE — If metric exists in table, ignore paragraph values
  ✅ FIX 34: TOTAL ROW DETECTION — Only accept rows containing total/grand total/overall or last row
  ✅ FIX 35: VALUE SCALE VALIDATION — Minimum thresholds: Scope1/2≥1000, Scope3≥10000, Water≥100000, Energy≥1e8 MJ
  ✅ FIX 36: STRICT UNIT VALIDATION — Waste→tonnes/kg, Emissions→tCO2e, Energy→MJ/kWh, Water→KL/m3
  ✅ FIX 37: PARAGRAPH FILTER — Reject if contains intervention/saved/reduced/initiative
  ✅ FIX 38: ESG SCORE HARD FILTER — Only if explicitly "ESG score" or "rating"
  ✅ FIX 39: CONFIDENCE CORRECTION — Table+total=0.9+, table only=0.7, paragraph max 0.6

All previous fixes (1-32) retained.

Usage:
    python evaluate_on_pdf.py --pdf_path "path/to/esg_report.pdf"
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import re
from dataclasses import dataclass, asdict
from collections import defaultdict
from difflib import SequenceMatcher

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

# 🎯 FIX 7: STRICT TARGET METRICS - Only extract these 7
TARGET_METRICS: Set[str] = {
    'SCOPE_1',
    'SCOPE_2', 
    'SCOPE_3',
    'ESG_SCORE',
    'WASTE_GENERATED',
    'ENERGY_CONSUMPTION',
    'WATER_USAGE'
}

# Unit Whitelist — ✅ FIX 12: Expanded with comprehensive ESG unit patterns
VALID_UNITS: Set[str] = {
    # Emissions
    "tCO2e", "Mt CO2e", "ktCO2e", "kgCO2", "kgCO2e", "tCO2", "tonnes CO2",
    "MtCO2e", "TCO2e",
    # Energy
    "GWh", "MWh", "kWh", "Mn kWh", "MJ", "GJ", "TJ", "PJ",
    "GJ/tonne", "MJ/unit", "kWh/unit",
    # Water
    "m³", "m3", "KL", "kilolitres", "kiloliters", "liters", "litres",
    "ML", "GL", "kl", "Ml",
    # Waste & Mass
    "kg", "tonnes", "t", "MT", "kt", "g",
    # Dimensionless / financial
    "%",
    "crore", "lakh",
}

# Unit normalization — ✅ FIX 12: Extended normalization table
UNIT_NORMALIZATION: Dict[str, str] = {
    # Emissions
    'tonnes': 'tCO2e', 'tons': 'tCO2e', 'tco2e': 'tCO2e', 'tco2': 'tCO2',
    'mtco2e': 'MtCO2e', 'million tonnes': 'Mt CO2e', 'mt co2e': 'Mt CO2e',
    'ktco2e': 'ktCO2e', 'kgco2': 'kgCO2', 'kgco2e': 'kgCO2e',
    'tonnes co2': 'tonnes CO2', 'tcO2e': 'tCO2e',
    # Energy
    'gwh': 'GWh', 'mwh': 'MWh', 'kwh': 'kWh',
    'mj': 'MJ', 'gj': 'GJ', 'tj': 'TJ', 'pj': 'PJ',
    'mn kwh': 'Mn kWh', 'million kwh': 'Mn kWh',
    'gj/tonne': 'GJ/tonne', 'mj/unit': 'MJ/unit', 'kwh/unit': 'kWh/unit',
    # Water
    'm3': 'm³', 'cubic meters': 'm³', 'cubic metres': 'm³',
    'kl': 'KL', 'kilolitres': 'KL', 'kiloliters': 'KL',
    'ml': 'ML', 'gl': 'GL',
    'liters': 'liters', 'litres': 'litres',
    # Waste & Mass
    '%': '%', 'percent': '%', 'percentage': '%',
    'mt': 'MT', 'metric tonnes': 'MT', 'metric tons': 'MT',
    'kt': 'kt', 'kg': 'kg', 't': 't', 'g': 'g',
    # Other
    'fte': 'FTE', 'employees': 'employees',
    'hours': 'hours', 'hours/employee': 'hours/employee',
    'crore': 'crore', 'lakh': 'lakh',
}

# ✅ FIX 36: STRICT UNIT COMPATIBILITY (replaces METRIC_UNIT_MAP for validation)
STRICT_UNIT_MAP: Dict[str, Set[str]] = {
    'SCOPE_1': {'tCO2e', 'Mt CO2e', 'MtCO2e', 'ktCO2e', 'tCO2', 'tonnes CO2', 'tonnes', 't', 'MT'},
    'SCOPE_2': {'tCO2e', 'Mt CO2e', 'MtCO2e', 'ktCO2e', 'tCO2', 'tonnes CO2', 'tonnes', 't', 'MT'},
    'SCOPE_3': {'tCO2e', 'Mt CO2e', 'MtCO2e', 'ktCO2e', 'tCO2', 'tonnes CO2', 'tonnes', 't', 'MT'},
    'ENERGY_CONSUMPTION': {'MJ', 'GJ', 'TJ', 'PJ', 'kWh', 'MWh', 'GWh', 'Mn kWh'},
    'WATER_USAGE': {'KL', 'm³', 'm3', 'ML', 'GL'},
    'WASTE_GENERATED': {'tonnes', 'MT', 'kt', 't', 'kg'},
    'ESG_SCORE': set(),  # unitless
}

# Metric-specific keywords
METRIC_KEYWORDS: Dict[str, List[str]] = {
    'SCOPE_1': ['emissions', 'co2', 'ghg', 'carbon', 'scope 1', 'scope1', 'direct emissions'],
    'SCOPE_2': ['emissions', 'co2', 'ghg', 'carbon', 'scope 2', 'scope2', 'indirect emissions'],
    'SCOPE_3': ['emissions', 'co2', 'ghg', 'carbon', 'scope 3', 'scope3', 'value chain'],
    'ENERGY_CONSUMPTION': ['energy', 'electricity', 'power', 'consumption', 'kwh', 'mwh', 'gwh'],
    'WATER_USAGE': ['water', 'withdrawal', 'discharge', 'consumption'],
    'WASTE_GENERATED': ['waste', 'generated', 'disposal', 'solid waste'],
    'ESG_SCORE': ['esg', 'score', 'rating', 'sustainability', 'index'],
}

UNITLESS_ALLOWED_METRICS: Set[str] = {'ESG_SCORE'}

TEMPORAL_KEYWORDS: List[str] = [
    'compared', 'previous', 'last year', 'prior year', 'preceding',
    'increase', 'decrease', 'growth', 'decline', 'change',
]

# FIX 4: Invalid context words for numeric extraction — matched as whole words only
INVALID_CONTEXT_WORDS: Set[str] = {
    'phone', 'email', 'cin', 'code', 'section', 'page', 'id', 'number',
    'tel', 'fax', 'contact', 'reference'
}
# Pre-compiled word-boundary pattern for whole-word matching
_INVALID_CONTEXT_RE = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in INVALID_CONTEXT_WORDS) + r')\b',
    re.IGNORECASE,
)

# FIX 4: Noise values to reject
NOISE_VALUES: Set[int] = {1, 2, 3, 4, 5, 10}

GENERIC_ENTITY_WORDS: Set[str] = {
    'total', 'number', 'energy', 'water', 'waste', 'carbon', 'scope',
    'paid', 'value', 'women', 'employee', 'os', 'km',
}

# ✅ FIX 15+22: Extended intensity / rejection phrase patterns.
INTENSITY_PATTERNS: List[re.Pattern] = [
    re.compile(
        r'(?<!\w)\bper\s+(employee|fte|worker|revenue|unit\b|tonne\b|kwh\b|mwh\b|gwh\b|'
        r'product|barrel|ton\b|kg\b|m2|sqft|capita|head)',
        re.IGNORECASE,
    ),
    re.compile(r'\bintensity\b', re.IGNORECASE),
    re.compile(r'\bratio\b', re.IGNORECASE),
    re.compile(r'\breduced?\s+by\b', re.IGNORECASE),
    re.compile(r'\bchange\s+of\b', re.IGNORECASE),
    re.compile(r'\brate\s+of\b', re.IGNORECASE),
    re.compile(r'\btarget\b', re.IGNORECASE),
    re.compile(r'\breduction\b', re.IGNORECASE),
    re.compile(r'\bprojection\b', re.IGNORECASE),
    re.compile(r'\bforecast\b', re.IGNORECASE),
    re.compile(
        r'[\d.]+\s*(?:tCO2e?|kg|MJ|GJ|kWh)\s*/\s*(?:unit|tonne|employee|revenue|kwh|mwh)',
        re.IGNORECASE,
    ),
]

# ✅ FIX 37: Paragraph filter keywords (reject these in text extraction)
PARAGRAPH_REJECT_KEYWORDS = re.compile(
    r'\b(intervention|saved|reduced|initiative)\b', re.IGNORECASE
)

# ✅ FIX 14: Scientific-notation string pattern
_SCI_NOTATION_RE = re.compile(r'\b\d+\.?\d*[eE][+-]?\d+\b')

# ✅ FIX 20: Year / column-header patterns for table header detection
_YEAR_HEADER_RE = re.compile(
    r'\b(?:FY\s*)?(?:20\d{2}|19\d{2})'   # 2023, FY2023, FY 2024
    r'|(?:FY\s*\d{2})\b',                  # FY23, FY 24
    re.IGNORECASE,
)

# ✅ FIX 21: Context-score weights for value selection
CTX_SCORE_WEIGHTS = {
    'total':       +3,
    'metric_kw':   +3,
    'unit_match':  +2,
    'intensity':   -3,
    'target_word': -2,
}
_TARGET_WORDS_RE = re.compile(
    r'\b(target|reduction|projection|forecast|plan(?:ned)?|goal|objective)\b',
    re.IGNORECASE,
)

# ✅ FIX 17: Canonical output unit per metric family.
CANONICAL_UNIT: Dict[str, str] = {
    'tCO2e':    'tCO2e', 'tCO2':   'tCO2e', 'TCO2e':   'tCO2e',
    'tonnes CO2': 'tCO2e', 'tonnes': 'tCO2e', 't': 'tCO2e',
    'MT':       'tCO2e', 'kt':    'tCO2e',
    'kgCO2':    'tCO2e', 'kgCO2e': 'tCO2e',
    'ktCO2e':   'ktCO2e', 'MtCO2e':   'MtCO2e', 'Mt CO2e':  'MtCO2e',
    'MJ':  'MJ', 'GJ': 'GJ', 'TJ': 'TJ', 'PJ': 'PJ',
    'kWh': 'kWh', 'MWh': 'MWh', 'GWh': 'GWh', 'Mn kWh': 'Mn kWh',
    'GJ/tonne': 'GJ/tonne', 'MJ/unit': 'MJ/unit', 'kWh/unit': 'kWh/unit',
    'm³': 'KL', 'm3': 'KL', 'KL': 'KL', 'kl': 'KL',
    'kilolitres': 'KL', 'kiloliters': 'KL',
    'ML': 'ML', 'GL': 'GL',
    'liters': 'KL', 'litres': 'KL',
    'kg': 'kg',
    '%': '%', 'crore': 'crore', 'lakh': 'lakh',
    'UNKNOWN': 'UNKNOWN',
}

KG_TO_TONNE_METRICS: Set[str] = {'SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'WASTE_GENERATED'}

# ✅ FIX 35: Minimum value thresholds (after unit normalization to base units)
MIN_VALUES = {
    'SCOPE_1': 1000,      # tonnes CO2e
    'SCOPE_2': 1000,
    'SCOPE_3': 10000,
    'WATER_USAGE': 100000,  # KL
    'ENERGY_CONSUMPTION': 100_000_000,  # MJ (1e8 MJ)
    'WASTE_GENERATED': 1000,  # tonnes
    'ESG_SCORE': 0,       # no minimum
}

# ✅ FIX 39: Confidence base values
CONFIDENCE_TABLE_TOTAL = 0.95
CONFIDENCE_TABLE_ONLY = 0.75
CONFIDENCE_PARAGRAPH_MAX = 0.60


# ============================================================================
# LAYER 1: PDF EXTRACTION (Text + Tables)
# ============================================================================

class PDFExtractor:
    """Extract text and tables from PDF using pdfplumber"""
    
    def __init__(self):
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
    
    def extract_text_and_tables(self, pdf_path: str) -> Dict:
        """Extract both full text and tables per page"""
        result = {'full_text': '', 'pages': [], 'tables': []}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Text extraction
                text = page.extract_text()
                if text:
                    result['pages'].append({
                        'page_number': page_num,
                        'text': text
                    })
                    result['full_text'] += text + '\n\n'
                
                # Table extraction (grid tables)
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        # Clean empty cells and strip whitespace
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [
                                cell.strip() if cell and isinstance(cell, str) else (cell if cell else '')
                                for cell in row
                            ]
                            if any(cell for cell in cleaned_row):
                                cleaned_table.append(cleaned_row)
                        if len(cleaned_table) > 1:  # at least header + one data row
                            result['tables'].append({
                                'page': page_num,
                                'table': cleaned_table,
                                'table_index': table_idx
                            })
        
        return result


# ============================================================================
# LAYER 1b: ENHANCED TABLE PARSER (pdfplumber tables)
# ============================================================================

class EnhancedTableParser:
    """
    Enhanced table parser using pdfplumber's extract_tables() output.
    Handles multi-line headers, year detection, metric mapping, value validation.
    ✅ FIX 34: Only total rows (or last row) are accepted.
    ✅ FIX 35: Scale validation applied.
    ✅ FIX 36: Strict unit validation.
    ✅ FIX 39: Confidence set based on total row.
    """
    
    # Comprehensive mapping for row labels → canonical metric (7 target metrics)
    METRIC_SYNONYMS = {
        'SCOPE_1': [
            'scope 1', 'scope1', 'direct emissions', 'scope i', 'scope one',
            'ghg scope 1', 'carbon scope 1', 'co2 scope 1', 'scope 1 emissions'
        ],
        'SCOPE_2': [
            'scope 2', 'scope2', 'indirect emissions', 'scope ii', 'scope two',
            'ghg scope 2', 'purchased electricity', 'scope 2 emissions'
        ],
        'SCOPE_3': [
            'scope 3', 'scope3', 'value chain emissions', 'scope iii', 'scope three',
            'other indirect emissions', 'supply chain emissions', 'scope 3 emissions'
        ],
        'ENERGY_CONSUMPTION': [
            'energy consumption', 'total energy', 'energy use', 'electricity consumption',
            'fuel consumption', 'primary energy', 'energy purchased'
        ],
        'WATER_USAGE': [
            'water usage', 'water consumption', 'water withdrawal', 'total water',
            'water use', 'fresh water', 'water intake'
        ],
        'WASTE_GENERATED': [
            'waste generated', 'total waste', 'solid waste', 'hazardous waste',
            'non-hazardous waste', 'waste disposal', 'waste produced'
        ],
        'ESG_SCORE': [
            'esg score', 'esg rating', 'sustainability score', 'sustainability rating',
            'esg index', 'moody\'s score', 's&p global score', 'esg risk rating'
        ]
    }
    
    # Patterns for years
    YEAR_PATTERN = re.compile(r'\b(?:FY\s*)?(?:20\d{2}|19\d{2}|\d{2})\b', re.IGNORECASE)
    
    # Patterns to detect unit rows
    UNIT_PATTERN = re.compile(
        r'\b(?:tCO2e|MtCO2e|ktCO2e|kgCO2e|GWh|MWh|kWh|m³|m3|KL|kl|'
        r'MT|tonnes|kg|%|MJ|GJ)\b', re.IGNORECASE
    )
    
    # Patterns to reject values (intensity, target, forecast)
    REJECT_PATTERNS = [
        re.compile(r'\bper\s+(employee|fte|unit|tonne|kwh|mwh|revenue|capita)', re.I),
        re.compile(r'\bintensity\b', re.I),
        re.compile(r'\btarget\b', re.I),
        re.compile(r'\breduction\b', re.I),
        re.compile(r'\bprojection\b', re.I),
        re.compile(r'\bforecast\b', re.I),
        re.compile(r'[\d.]+\s*/\s*(employee|unit|tonne|kwh)', re.I),
    ]
    
    @classmethod
    def parse_tables(cls, tables_data: List[Dict], page_texts: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Parse all tables extracted from PDF.
        If no grid tables yield results, fall back to text-block parser.
        """
        all_metrics = []
        if tables_data:
            for table_info in tables_data:
                metrics = cls._parse_single_table(table_info['table'], table_info['page'])
                all_metrics.extend(metrics)
        
        # Fallback: if no metrics from grid tables and page texts provided, use legacy block parser
        if not all_metrics and page_texts:
            legacy_parser = TableParserLegacy()
            for page in page_texts:
                rows = legacy_parser.parse_page(page['text'], page['page_number'])
                all_metrics.extend(rows)
        
        return all_metrics
    
    @classmethod
    def _parse_single_table(cls, table: List[List[str]], page_num: int) -> List[Dict]:
        """Parse one extracted table (list of rows)"""
        if not table or len(table) < 2:
            return []
        
        # Step 1: Detect header rows (may be multiple)
        header_rows, data_start_idx = cls._detect_header_rows(table)
        if not header_rows:
            return []
        
        # Step 2: Determine which column contains the latest year
        year_col_idx = cls._find_latest_year_column(header_rows)
        
        # Step 3: Extract units (from header rows or from column data)
        col_units = cls._extract_column_units(header_rows, table, data_start_idx)
        
        # Step 4: Process each data row
        results = []
        num_rows = len(table)
        for row_idx in range(data_start_idx, num_rows):
            row = table[row_idx]
            if not row or len(row) < 2:
                continue
            
            # First column is metric label
            metric_label = row[0].strip() if row[0] else ''
            if not metric_label:
                continue
            
            # ✅ FIX 34: Total row detection
            if not cls._is_total_row(metric_label, row_idx, num_rows):
                continue
            
            # Map label to canonical metric
            canonical_metric = cls._map_metric(metric_label)
            if not canonical_metric:
                continue
            
            # Get the value from the latest year column
            if year_col_idx >= len(row):
                continue
            value_cell = row[year_col_idx].strip() if row[year_col_idx] else ''
            if not value_cell:
                continue
            
            # Extract numeric value and unit from the cell
            value, unit = cls._extract_value_and_unit(value_cell, col_units[year_col_idx])
            if value is None:
                continue
            
            # Apply unit normalization and conversion
            unit, value = ValueExtractor.canonicalize_unit(unit, canonical_metric, value)
            
            # ✅ FIX 36: Strict unit validation
            if canonical_metric in STRICT_UNIT_MAP and STRICT_UNIT_MAP[canonical_metric]:
                if unit not in STRICT_UNIT_MAP[canonical_metric]:
                    continue
            
            # ✅ FIX 35: Value scale validation
            min_val = MIN_VALUES.get(canonical_metric, 0)
            if value < min_val:
                continue
            
            # Reject if context (row or header) indicates intensity/target
            context_str = ' '.join(row[:3]) + ' ' + ' '.join(header_rows[0][:3])
            if cls._is_rejected_context(context_str):
                continue
            
            # ✅ FIX 39: Confidence based on total row and table source
            confidence = CONFIDENCE_TABLE_TOTAL if cls._is_total_row(metric_label, row_idx, num_rows) else CONFIDENCE_TABLE_ONLY
            
            # Build result
            results.append({
                'normalized_metric': canonical_metric,
                'value': value,
                'unit': unit,
                'entity_text': metric_label,
                'context': context_str[:200],
                'section_type': 'Environmental',
                'confidence': confidence,
                'validation_status': 'VALID',
                'validation_issues': [],
                'source_type': 'table',
                'page': page_num,
            })
        
        return results
    
    @classmethod
    def _is_total_row(cls, label: str, row_idx: int, total_rows: int) -> bool:
        """✅ FIX 34: Return True if row contains total keyword or is last row."""
        label_lower = label.lower()
        if re.search(r'\b(total|grand total|overall)\b', label_lower):
            return True
        # Last row of table (assuming summary row)
        if row_idx == total_rows - 1:
            return True
        return False
    
    @classmethod
    def _detect_header_rows(cls, table: List[List[str]]) -> Tuple[List[List[str]], int]:
        """Detect header rows (may be one or two rows). Returns (header_rows, first_data_row_index)."""
        header_rows = []
        data_start = 0
        
        for i, row in enumerate(table):
            row_text = ' '.join([c for c in row if c]).lower()
            has_year = bool(cls.YEAR_PATTERN.search(row_text))
            has_unit = bool(cls.UNIT_PATTERN.search(row_text))
            numeric_cols = sum(1 for cell in row if re.search(r'\d', cell))
            
            if (has_year or has_unit) and numeric_cols <= len(row) // 2:
                header_rows.append(row)
            else:
                if header_rows:
                    data_start = i
                    break
                if i == 0 and not header_rows:
                    return [], 0
        
        if not header_rows:
            if len(table) > 1:
                header_rows = [table[0]]
                data_start = 1
            else:
                return [], 0
        
        return header_rows, data_start
    
    @classmethod
    def _find_latest_year_column(cls, header_rows: List[List[str]]) -> int:
        """Find column index containing the most recent year."""
        best_col = -1
        latest_year = 0

        for row in header_rows:
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                years = cls.YEAR_PATTERN.findall(cell)
                for y_str in years:
                    # Extract only the numeric part (e.g., from "FY 2025" -> "2025")
                    year_digits = re.search(r'\d+', y_str)
                    if not year_digits:
                        continue
                    year_val = int(year_digits.group())
                    # Convert 2-digit year to 4-digit (assuming 2000+)
                    if year_val < 100:
                        year_val = 2000 + year_val if year_val >= 20 else 2000 + year_val
                    if year_val > latest_year:
                        latest_year = year_val
                        best_col = col_idx
        return best_col
    
    @classmethod
    def _extract_column_units(cls, header_rows: List[List[str]], table: List[List[str]], data_start: int) -> List[str]:
        """Determine unit for each column from headers or first data row."""
        num_cols = max(len(row) for row in table) if table else 0
        units = [''] * num_cols
        
        # Try header rows
        for row in header_rows:
            for col_idx, cell in enumerate(row):
                if col_idx < len(units) and cell:
                    unit_match = cls.UNIT_PATTERN.search(cell)
                    if unit_match:
                        units[col_idx] = unit_match.group(0).strip()
        
        # If missing, sample first data row
        if data_start < len(table):
            sample_row = table[data_start]
            for col_idx, cell in enumerate(sample_row):
                if col_idx < len(units) and not units[col_idx] and cell:
                    _, unit = cls._extract_value_and_unit(cell, '')
                    if unit:
                        units[col_idx] = unit
        
        return units
    
    @classmethod
    def _extract_value_and_unit(cls, cell: str, fallback_unit: str = '') -> Tuple[Optional[float], str]:
        """Extract numeric value and unit from a cell string."""
        cell_clean = cell.replace(',', '').strip()
        match = re.search(r'([\d.]+)\s*([a-zA-Z%³²]+)?', cell_clean)
        if not match:
            return None, ''
        try:
            value = float(match.group(1))
        except ValueError:
            return None, ''
        unit = match.group(2) if match.group(2) else ''
        if unit:
            unit = unit.strip()
        elif fallback_unit:
            unit = fallback_unit
        if unit:
            unit_lower = unit.lower()
            unit = UNIT_NORMALIZATION.get(unit_lower, unit)
        if value < 1 and unit != '%':
            return None, ''
        return value, unit
    
    @classmethod
    def _map_metric(cls, label: str) -> Optional[str]:
        """Map row label to one of 7 target metrics using fuzzy matching."""
        label_lower = label.lower().strip()
        label_lower = re.sub(r'[^\w\s]', '', label_lower)
        best_match = None
        best_score = 0.0
        for metric, synonyms in cls.METRIC_SYNONYMS.items():
            if label_lower in synonyms:
                return metric
            for syn in synonyms:
                if syn in label_lower:
                    score = len(syn) / len(label_lower) if label_lower else 0
                    if score > best_score:
                        best_score = score
                        best_match = metric
        if best_score > 0.6:
            return best_match
        return None
    
    @classmethod
    def _is_rejected_context(cls, context: str) -> bool:
        """Check if context contains intensity/target/forecast indicators."""
        context_lower = context.lower()
        for pattern in cls.REJECT_PATTERNS:
            if pattern.search(context_lower):
                return True
        return False


# ============================================================================
# LEGACY TABLE PARSER (Fallback for non-grid tables)
# ============================================================================

class TableParserLegacy:
    """Original text-block table parser (used as fallback) - also updated with total-row and scale validation"""
    
    def parse_page(self, page_text: str, page_number: int) -> List[Dict]:
        results = []
        blocks = re.split(r'\n\s*\n', page_text)
        for block in blocks:
            if self._is_table_block(block):
                rows = self._extract_table_rows(block, page_number)
                results.extend(rows)
        return results
    
    def _is_table_block(self, block: str) -> bool:
        lines_with_numbers = sum(1 for line in block.splitlines() if re.search(r'\b\d[\d,]*\.?\d*\b', line))
        return lines_with_numbers >= 3
    
    def _extract_table_rows(self, block: str, page_number: int) -> List[Dict]:
        lines = block.splitlines()
        if len(lines) < 2:
            return []
        header_idx, latest_col = self._detect_header_and_latest_col(lines)
        results = []
        header_line = lines[header_idx] if header_idx < len(lines) else ''
        header_unit = self._extract_unit_from_header(header_line)
        total_rows = len(lines)
        for row_idx, line in enumerate(lines[header_idx + 1:], start=header_idx+1):
            row = self._parse_data_row(line, latest_col, header_unit, page_number, row_idx, total_rows)
            if row:
                results.append(row)
        return results
    
    def _detect_header_and_latest_col(self, lines: List[str]) -> Tuple[int, int]:
        for i, line in enumerate(lines[:5]):
            year_matches = list(_YEAR_HEADER_RE.finditer(line))
            if year_matches:
                best_match = self._latest_year_match(year_matches, line)
                col_idx = self._match_to_col_index(line, best_match)
                return i, col_idx
            if _UNIT_PATTERNS_RE.search(line):
                return i, -1
        return 0, -1
    
    def _latest_year_match(self, matches: List[re.Match], line: str) -> re.Match:
        def year_value(m: re.Match) -> int:
            text = m.group().upper().replace('FY', '').replace(' ', '')
            try:
                y = int(text)
                return y + 2000 if y < 100 else y
            except ValueError:
                return 0
        return max(matches, key=year_value)
    
    def _match_to_col_index(self, line: str, match: re.Match) -> int:
        prefix = line[:match.start()]
        cols = re.split(r'\t|  +', prefix)
        return len(cols)
    
    def _extract_unit_from_header(self, header_line: str) -> str:
        m = _UNIT_PATTERNS_RE.search(header_line)
        if m:
            raw = m.group(1).strip()
            norm = UNIT_NORMALIZATION.get(raw.lower(), raw)
            if norm in VALID_UNITS:
                return norm
        return ''
    
    def _parse_data_row(self, line: str, latest_col: int, fallback_unit: str, page_number: int, row_idx: int, total_rows: int) -> Optional[Dict]:
        if not line.strip():
            return None
        cols = re.split(r'\t|  +', line.strip())
        if not cols:
            return None
        metric_label = cols[0].strip()
        metric_name = self._map_metric(metric_label)
        if not metric_name:
            return None
        
        # ✅ FIX 34: Total row detection
        if not self._is_total_row(metric_label, row_idx, total_rows):
            return None
        
        numeric_cols = []
        for idx, col in enumerate(cols[1:], start=1):
            col = col.strip()
            if any(p.search(col) for p in INTENSITY_PATTERNS):
                continue
            m = re.search(r'([\d,]+\.?\d*)', col)
            if not m:
                continue
            try:
                val = float(m.group(1).replace(',', ''))
            except ValueError:
                continue
            if val < 1 or (1900 <= val <= 2030) or val > 1e9:
                continue
            unit_m = _UNIT_PATTERNS_RE.search(col)
            unit = ''
            if unit_m:
                raw = unit_m.group(1).strip()
                unit = UNIT_NORMALIZATION.get(raw.lower(), raw)
                if unit not in VALID_UNITS:
                    unit = ''
            if not unit:
                unit = fallback_unit
            numeric_cols.append((idx, val, unit))
        if not numeric_cols:
            return None
        if latest_col > 0 and numeric_cols:
            max_col = numeric_cols[-1][0]
            effective_col = min(latest_col, max_col)
            chosen = min(numeric_cols, key=lambda x: abs(x[0] - effective_col))
        else:
            chosen = numeric_cols[-1]
        _, value, unit = chosen
        unit, value = ValueExtractor.canonicalize_unit(unit, metric_name, value)
        
        # ✅ FIX 36: Strict unit validation
        if metric_name in STRICT_UNIT_MAP and STRICT_UNIT_MAP[metric_name]:
            if unit not in STRICT_UNIT_MAP[metric_name]:
                return None
        
        # ✅ FIX 35: Scale validation
        min_val = MIN_VALUES.get(metric_name, 0)
        if value < min_val:
            return None
        
        # ✅ FIX 39: Confidence based on total row and table source
        confidence = CONFIDENCE_TABLE_TOTAL if self._is_total_row(metric_label, row_idx, total_rows) else CONFIDENCE_TABLE_ONLY
        
        return {
            'normalized_metric': metric_name,
            'value': value,
            'unit': unit,
            'entity_text': metric_label,
            'context': line[:200],
            'section_type': 'Environmental',
            'confidence': confidence,
            'validation_status': 'VALID',
            'validation_issues': [],
            'source_type': 'table',
            'page': page_number,
        }
    
    def _is_total_row(self, label: str, row_idx: int, total_rows: int) -> bool:
        """✅ FIX 34: Return True if row contains total keyword or is last row."""
        label_lower = label.lower()
        if re.search(r'\b(total|grand total|overall)\b', label_lower):
            return True
        # Last row of table (assuming summary row)
        if row_idx == total_rows - 1:
            return True
        return False
    
    def _map_metric(self, label: str) -> Optional[str]:
        label_lower = label.lower().strip()
        for key, synonyms in EnhancedTableParser.METRIC_SYNONYMS.items():
            for syn in synonyms:
                if syn in label_lower:
                    return key
        return None


# ============================================================================
# LAYER 2-3: TEXT PREPROCESSOR (unchanged)
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
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'\n\d+\n', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[Dict]:
        """Chunk text into overlapping segments"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
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
        """Classify text section"""
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
# LAYER 4: ESG CANDIDATE FILTER (unchanged)
# ============================================================================

class ESGCandidateFilter:
    """
    ✅ FIX 3: ESG FILTER TIGHTENING
    NEW RULES:
    - Minimum 2 ESG keywords (unchanged)
    - MUST have at least one numeric value (NEW)
    - Confidence threshold >= 0.4 (unchanged)
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
            print("⚠️  ESG filter model not found. Using keyword-based filter (FIX 3 applied).")
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
        ✅ FIX 3: TIGHTENED KEYWORD FILTERING
        Requirements:
        1. >= 2 ESG keywords
        2. Has at least one numeric value (NEW)
        3. Confidence >= 0.4
        """
        text_lower = text.lower()
        keyword_count = sum(1 for kw in self.esg_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        # FIX 3: MUST have numeric value
        has_number = bool(re.search(r'\d+\.?\d*', text))
        
        # FIX 3: Both conditions required
        if keyword_count < 2 or not has_number:
            return False, 0.0
        
        # Calculate confidence
        base_confidence = min(keyword_count / 5.0, 1.0)
        
        # Apply confidence threshold
        if base_confidence < 0.4:
            return False, base_confidence
        
        if negative_count > 0:
            base_confidence *= max(0.5, 1.0 - negative_count * 0.15)
        
        return True, base_confidence


# ============================================================================
# LAYER 5: NER MODEL (unchanged)
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
                current_entity = {
                    'text': text[offset_mapping[idx][0]:offset_mapping[idx][1]],
                    'start': offset_mapping[idx][0].item(),
                    'end': offset_mapping[idx][1].item(),
                    'label': label[2:],
                    'confidence': confidence,
                    'section_type': section_type
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] = text[current_entity['start']:offset_mapping[idx][1]]
                current_entity['end'] = offset_mapping[idx][1].item()
                current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
        
        if current_entity:
            entities.append(current_entity)
        
        return entities


# ============================================================================
# LAYER 6: METRIC CLASSIFIER (unchanged)
# ============================================================================

class MetricClassifier:
    """
    ✅ FIX 1: CLASSIFIER CONFIDENCE GATING
    NEW RULE:
    - If classification confidence < 0.7 → Return "UNKNOWN"
    - Do NOT allow further processing for weak predictions
    """
    
    def __init__(self, model_path: str):
        print(f"Loading Metric Classifier from {model_path}...")
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
        
        print(f"✓ Classifier loaded with {len(self.id2label)} classes (FIX 1: confidence gating @ 0.7)")
    
    def classify(self, metric_text: str, context: str, section_type: str = "") -> Tuple[str, float]:
        """
        ✅ FIX 1: Classify with HARD CONFIDENCE THRESHOLD
        Returns UNKNOWN if confidence < 0.7
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
        
        # ✅ FIX 1: CONFIDENCE GATING
        if confidence < 0.7:
            return "UNKNOWN", confidence
        
        # ✅ FIX 7: METRIC ENFORCEMENT
        if normalized_metric not in TARGET_METRICS:
            return "UNKNOWN", 0.0
        
        return normalized_metric, confidence


# ============================================================================
# LAYER 7-8: VALUE EXTRACTOR (unchanged, but used by legacy parser)
# ============================================================================

_UNIT_PATTERNS_RE = re.compile(
    r'\b('
    r'tCO2e|MtCO2e|ktCO2e|kgCO2e|kgCO2|tCO2|tonnes\s+CO2|Mt\s+CO2e|'
    r'TCO2e|'
    r'GWh|MWh|kWh|Mn\s+kWh|GJ/tonne|MJ/unit|kWh/unit|TJ|PJ|GJ|MJ|'
    r'm³|m3|kiloliters|kilolitres|KL|kl|ML|GL|liters|litres|'
    r'MT|kt\b|tonnes\b|tonne\b|tons\b|ton\b|kg\b|'
    r'%'
    r')',
    re.IGNORECASE,
)


class ValueExtractor:
    """
    Extract values and units with multi-stage fallback.
    Stage 1 – Direct adjacent match.
    Stage 2 – Window search: ±UNIT_WINDOW chars around the number.
    Stage 3 – Table-aware: if context has 3+ numbers, scan nearest unit.
    Stage 4 – Fallback unit UNKNOWN when value is strong & metric confident.
    """

    UNIT_WINDOW = 50

    def __init__(self):
        # Ordered from most to least specific so the first match wins
        self.patterns = [
            (r'([\d,]+\.?\d*)\s*(%)', 'pct'),
            (r'([\d,]+\.\d+)\s+([a-zA-Z/%³]+(?:\s+[a-zA-Z²³]+)?)', 'full_float'),
            (r'([\d,]+)\s+([a-zA-Z/%³]+(?:\s+[a-zA-Z²³]+)?)', 'full_int'),
            (r'([\d,]+\.\d+)', 'float_only'),
            (r'([\d,]+)', 'int_only'),
        ]

        self.confidence_map = {
            'pct':        0.95,
            'full_float': 0.90,
            'full_int':   0.85,
            'float_only': 0.60,
            'int_only':   0.50,
        }

    def extract_context(self, text: str, entity_start: int, entity_end: int,
                        window: int = 150) -> str:
        """Return a sentence-aware context window around the entity."""
        sentence_context = self._extract_sentence_context(text, entity_start, entity_end)
        if sentence_context and len(sentence_context) >= 30:
            return sentence_context
        context_start = max(0, entity_start - window)
        context_end   = min(len(text), entity_end + window)
        return text[context_start:context_end]

    def extract_value(
        self,
        context: str,
        entity_start_in_context: Optional[int] = None,
        entity_end_in_context: Optional[int] = None,
        classifier_confidence: float = 0.0,
    ) -> Optional[Dict]:
        """
        Multi-stage value + unit extraction.
        Returns a dict with keys: value, unit, confidence, num_candidates
        """
        # ✅ FIX 15: Reject the entire context if it is clearly an intensity metric
        if self._is_intensity_context(context):
            return None

        # Stage 1: extract raw candidates via regex
        raw_candidates = self._extract_all_candidates(context)

        if not raw_candidates:
            return None

        # ✅ FIX 4 + FIX 14: Hard numeric filter (includes tiny-value rejection)
        candidates = [c for c in raw_candidates if self._is_valid_value(c, context)]

        if not candidates:
            return None

        # ✅ FIX 9 + FIX 10: For candidates that have no unit yet, try window search
        candidates = self._fill_units_via_window(candidates, context)

        # ✅ FIX 10: Table-aware pass — if many numbers in context, re-associate
        if self._is_tabular_context(context):
            candidates = self._table_aware_unit_match(candidates, context)

        # ✅ FIX 21: CONTEXT-SCORE BASED VALUE SELECTION
        scored = self._score_candidates(candidates, context)
        if scored:
            candidates = scored

        best          = candidates[0]
        num_candidates = len(candidates)

        # ✅ FIX 11: Fallback — if still no unit but value is strong, mark UNKNOWN
        if not best['unit'] and best['confidence'] >= 0.5 and classifier_confidence >= 0.8:
            best = dict(best)
            best['unit']       = 'UNKNOWN'
            best['confidence'] *= 0.7

        result = {
            'value':          best['value'],
            'unit':           best['unit'],
            'confidence':     best['confidence'],
            'num_candidates': num_candidates,
        }

        if num_candidates > 1:
            result['confidence'] *= 0.6

        return result

    # ------------------------------------------------------------------
    # FIX 21: Context-score based value selection
    # ------------------------------------------------------------------

    def _score_candidates(self, candidates: List[Dict], context: str) -> List[Dict]:
        """
        ✅ FIX 21: Score each candidate using context signals.
        Score = +3 "total" in local window
              + +3 any METRIC_KEYWORDS hit in context
              + +2 unit is compatible with any target metric
              + -3 any INTENSITY_PATTERNS hit
              + -2 any _TARGET_WORDS_RE hit
        Sort DESC by score; if all scores equal, tiebreak by value DESC.
        """
        context_lower = context.lower()
        ctx_has_intensity = any(p.search(context) for p in INTENSITY_PATTERNS)
        ctx_has_target    = bool(_TARGET_WORDS_RE.search(context))

        # Check all metric keywords at context level
        ctx_metric_kw = any(
            any(kw in context_lower for kw in kws)
            for kws in METRIC_KEYWORDS.values()
        )

        for c in candidates:
            score = 0
            # Local window around the number (±80 chars)
            lo   = max(0, c['match_start'] - 80)
            hi   = min(len(context), c['match_end'] + 80)
            local = context[lo:hi].lower()

            if 'total' in local:
                score += CTX_SCORE_WEIGHTS['total']
            if ctx_metric_kw:
                score += CTX_SCORE_WEIGHTS['metric_kw']
            if c['unit'] and c['unit'] != 'UNKNOWN':
                # Check if unit matches any metric family
                for valid_set in STRICT_UNIT_MAP.values():
                    if c['unit'] in valid_set:
                        score += CTX_SCORE_WEIGHTS['unit_match']
                        break
            if ctx_has_intensity:
                score += CTX_SCORE_WEIGHTS['intensity']   # negative
            if ctx_has_target:
                score += CTX_SCORE_WEIGHTS['target_word'] # negative

            c['_ctx_score'] = score

        # Sort: score DESC, then value DESC as tiebreaker
        candidates_sorted = sorted(
            candidates,
            key=lambda c: (c['_ctx_score'], c['value']),
            reverse=True,
        )
        return candidates_sorted

    # ------------------------------------------------------------------
    # FIX 15: Intensity context detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_intensity_context(context: str) -> bool:
        """
        ✅ FIX 15+22: Return True if the context describes an intensity,
        target, projection or forecast metric.
        """
        for pat in INTENSITY_PATTERNS:
            if pat.search(context):
                return True
        return False

    # ------------------------------------------------------------------
    # FIX 9: Window-based unit search
    # ------------------------------------------------------------------

    def _fill_units_via_window(self, candidates: List[Dict], context: str) -> List[Dict]:
        """
        ✅ FIX 9: For candidates with no unit, scan ±UNIT_WINDOW chars for a unit token.
        """
        result = []
        for c in candidates:
            if c['unit']:
                result.append(c)
                continue

            lo = max(0, c['match_start'] - self.UNIT_WINDOW)
            hi = min(len(context), c['match_end'] + self.UNIT_WINDOW)
            window_text = context[lo:hi]

            unit_match = _UNIT_PATTERNS_RE.search(window_text)
            if unit_match:
                raw_unit = unit_match.group(1).strip()
                normalised = self._normalize_unit(raw_unit)
                if self._is_valid_unit(normalised):
                    c = dict(c)           # copy before mutating
                    c['unit'] = normalised

            result.append(c)

        return result

    # ------------------------------------------------------------------
    # FIX 10: Table-aware extraction
    # ------------------------------------------------------------------

    def _is_tabular_context(self, context: str) -> bool:
        """Heuristic: 3+ distinct numbers → probably a table row/section."""
        numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', context)
        return len(numbers) >= 3

    def _table_aware_unit_match(self, candidates: List[Dict],
                                context: str) -> List[Dict]:
        """
        ✅ FIX 10: In tabular contexts try to find the *nearest* unit for
        each unit-less number, scanning the whole context rather than
        just the local window, and prefer units that appear *after* the number
        (column header → value layout is more common than value → header).
        """
        all_unit_spans = [
            (m.start(), m.end(), self._normalize_unit(m.group(1)))
            for m in _UNIT_PATTERNS_RE.finditer(context)
            if self._is_valid_unit(self._normalize_unit(m.group(1)))
        ]

        if not all_unit_spans:
            return candidates

        updated = []
        for c in candidates:
            if c['unit']:
                updated.append(c)
                continue

            num_mid = (c['match_start'] + c['match_end']) / 2
            best_span = min(
                all_unit_spans,
                key=lambda s: abs((s[0] + s[1]) / 2 - num_mid)
            )
            c = dict(c)
            c['unit'] = best_span[2]
            updated.append(c)

        return updated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_sentence_context(self, text: str, entity_start: int,
                                   entity_end: int) -> Optional[str]:
        sentence_boundaries = list(re.finditer(r'[.!?]\s+', text))
        if not sentence_boundaries:
            return None

        sentences, prev_end = [], 0
        for match in sentence_boundaries:
            sentences.append((prev_end, match.end()))
            prev_end = match.end()
        if prev_end < len(text):
            sentences.append((prev_end, len(text)))

        for i, (s_start, s_end) in enumerate(sentences):
            if s_start <= entity_start < s_end:
                end_idx = min(i + 1, len(sentences) - 1)
                return text[sentences[i][0]:sentences[end_idx][1]].strip()
        return None

    def _is_valid_value(self, candidate: Dict, context: str) -> bool:
        """
        ✅ FIX 4:  HARD NUMERIC FILTERING
        ✅ FIX 14: VALUE PRIORITIZATION — reject tiny/scientific-notation values
        """
        value = candidate['value']
        unit  = candidate['unit']

        # FIX 14+23: reject sub-unit decimals (< 1) — these are always factors/intensities
        if value < 1 and unit != '%':
            return False

        if value > 1_000_000_000:
            return False
        if value in NOISE_VALUES and unit != '%':
            return False
        if 1900 <= value <= 2030:
            return False
        # FIX 4: whole-word check for invalid context keywords
        if _INVALID_CONTEXT_RE.search(context):
            return False
        return True

    def _extract_all_candidates(self, context: str) -> List[Dict]:
        """Extract all (value, unit) candidate pairs from context."""
        candidates  = []
        seen_spans  = set()

        # ✅ FIX 14: Pre-mark spans that are scientific-notation strings so we can skip them
        sci_spans = {(m.start(), m.end()) for m in _SCI_NOTATION_RE.finditer(context)}

        for pattern, pattern_type in self.patterns:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                span = (match.start(), match.end())
                if any(self._spans_overlap(span, s) for s in seen_spans):
                    continue

                # ✅ FIX 14: skip if the number token overlaps a sci-notation string
                if any(self._spans_overlap((match.start(), match.start() + len(match.group(1))), s)
                       for s in sci_spans):
                    continue

                try:
                    value = float(match.group(1).replace(',', ''))
                except (ValueError, IndexError):
                    continue

                unit = ""
                if len(match.groups()) >= 2 and match.group(2):
                    unit_raw = match.group(2).strip()
                    unit = self._normalize_unit(unit_raw)

                if unit and not self._is_valid_unit(unit):
                    unit = ""

                candidates.append({
                    'value':      value,
                    'unit':       unit,
                    'confidence': self.confidence_map[pattern_type],
                    'match_start': match.start(),
                    'match_end':   match.end(),
                })
                seen_spans.add(span)

        return candidates

    def _normalize_unit(self, unit_raw: str) -> str:
        unit_lower = unit_raw.lower().strip()
        return UNIT_NORMALIZATION.get(unit_lower, unit_raw)

    @staticmethod
    def canonicalize_unit(unit: str, metric: str, value: float) -> Tuple[str, float]:
        """
        ✅ FIX 17: Convert extracted unit+value to canonical output form.
        Rules:
          kg  → tCO2e  (value ÷ 1000)   for emissions/waste metrics
          m³  → KL     (1:1)
          liters/litres → KL (÷ 1000)
          All others: mapped via CANONICAL_UNIT table (value unchanged).
        Returns (canonical_unit, adjusted_value).
        """
        if not unit or unit == 'UNKNOWN':
            return unit, value

        # kg → tonnes for relevant metrics
        if unit == 'kg' and metric in KG_TO_TONNE_METRICS:
            return 'tCO2e', round(value / 1000.0, 6)

        # liters / litres → KL
        if unit in ('liters', 'litres'):
            return 'KL', round(value / 1000.0, 6)

        canonical = CANONICAL_UNIT.get(unit, unit)
        return canonical, value

    def _is_valid_unit(self, unit: str) -> bool:
        return unit in VALID_UNITS

    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])


# ============================================================================
# LAYER 11: VALIDATION (with ESG hard filter)
# ============================================================================

class MetricValidator:
    """
    ✅ FIX 5: ESG_SCORE STRICT VALIDATION
    NEW RULES for ESG_SCORE:
    - Value must be between 20-100
    - Context must contain score keywords: ["score", "rating", "index"]
    - Reject if either condition fails
    ✅ FIX 38: ESG_SCORE hard filter (already done, but reinforced)
    """
    
    def __init__(self):
        self.value_ranges = {
            'SCOPE_1': (0, 10_000_000),
            'SCOPE_2': (0, 10_000_000),
            'SCOPE_3': (0, 50_000_000),
            'ENERGY_CONSUMPTION': (0, 100_000_000),
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
        Validate extracted metric.
        Returns: (status, issues, penalty)
        - penalty = 0.0 means complete discard
        """
        issues = []
        status = "VALID"
        penalty = 1.0
        
        # ✅ FIX 5 + FIX 38: ESG_SCORE STRICT VALIDATION
        if metric == 'ESG_SCORE':
            # Rule 1: Value range 20-100
            if value < 20 or value > 100:
                issues.append(f"ESG_SCORE value {value} outside valid range [20, 100]")
                status = "INVALID"
                penalty = 0.0
                return status, issues, penalty
            
            # Rule 2: Must have score keyword in context
            context_lower = context.lower()
            score_keywords = ['score', 'rating', 'index']
            has_score_keyword = any(kw in context_lower for kw in score_keywords)
            
            if not has_score_keyword:
                issues.append("ESG_SCORE requires score-related keyword in context")
                status = "INVALID"
                penalty = 0.0
                return status, issues, penalty
        
        # Range validation for other metrics
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
        
        # ✅ FIX 36: Metric-unit compatibility (strict)
        if metric in STRICT_UNIT_MAP:
            valid_units_for_metric = STRICT_UNIT_MAP[metric]

            if valid_units_for_metric and unit and unit != 'UNKNOWN':
                if unit not in valid_units_for_metric:
                    issues.append(f"Unit '{unit}' incompatible with metric '{metric}'")
                    status = "INVALID"
                    penalty = 0.0
        
        # ✅ FIX 13: NO EARLY DISCARD ON MISSING UNIT
        if not unit:
            if metric not in UNITLESS_ALLOWED_METRICS:
                issues.append(f"Missing unit for metric '{metric}'")
                status = "INVALID"
                penalty = 0.0
        elif unit == 'UNKNOWN':
            if metric not in UNITLESS_ALLOWED_METRICS:
                issues.append(f"Unit unknown for metric '{metric}' — proceeding with penalty")
                if status == "VALID":
                    status = "WARNING"
                penalty *= 0.6
        
        # Keyword validation
        keyword_penalty = self._check_metric_keywords(metric, context, entity_text)
        if keyword_penalty < 1.0:
            if keyword_penalty == 0.0:
                issues.append(f"No domain keywords found for metric '{metric}'")
                status = "INVALID"
            else:
                issues.append(f"Weak keyword match for metric '{metric}'")
                if status == "VALID":
                    status = "WARNING"
            penalty *= keyword_penalty
        
        return status, issues, penalty
    
    def _check_metric_keywords(self, metric: str, context: str, entity_text: str) -> float:
        """Check if context contains domain keywords"""
        if metric not in METRIC_KEYWORDS:
            return 1.0
        
        required_keywords = METRIC_KEYWORDS[metric]
        combined_text = f"{entity_text} {context}".lower()
        
        matches = sum(1 for kw in required_keywords if kw in combined_text)
        
        if matches >= 2:
            return 1.0
        elif matches == 1:
            return 0.7
        else:
            return 0.0


# ============================================================================
# LAYER 10: CONFIDENCE SCORING (updated for paragraph max cap)
# ============================================================================

class ConfidenceScorer:
    """
    ✅ FIX 6: CONFIDENCE SCORING CORRECTION
    NEW RULES:
    - Confidence range: [0.1, 0.95] (was [0.3, 0.95])
    - Discard if confidence < 0.5
    ✅ FIX 39: Paragraph max cap = 0.60, Table+total = 0.95, Table only = 0.75
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
        metric: str = "",
        source_type: str = "paragraph",
    ) -> float:
        """
        ✅ FIX 6: Calculate confidence with corrected range [0.1, 0.95]
        ✅ FIX 39: Table-source bonus is replaced by explicit caps
        """
        # Base weighted average
        base_score = (
            ner_confidence * 0.4 +
            classification_confidence * 0.3 +
            value_confidence * 0.3
        )

        # Apply penalties
        if not unit:
            base_score *= 0.7

        base_score *= validation_penalty

        if num_value_candidates > 1:
            base_score *= 0.6

        if len(entity_text.strip()) < 3:
            base_score *= 0.6

        if entity_text.strip().lower() in GENERIC_ENTITY_WORDS:
            base_score *= 0.7

        if not keyword_match:
            base_score *= 0.5

        # ESG_SCORE penalty
        if metric == 'ESG_SCORE':
            base_score *= 0.7

        # ✅ FIX 39: Apply caps based on source_type
        if source_type == 'paragraph':
            base_score = min(base_score, CONFIDENCE_PARAGRAPH_MAX)
        elif source_type == 'table':
            # For table, we do not cap upward; instead we set a minimum if it's a total row
            # but the caller (table parser) already sets confidence directly.
            # Here we just ensure it doesn't exceed 0.95
            base_score = max(base_score, 0.5)  # table values should be at least 0.5
        else:
            # fallback
            pass

        # ✅ FIX 6: Corrected confidence range [0.1, 0.95]
        confidence = max(0.1, min(0.95, base_score))

        return round(confidence, 4)


# ============================================================================
# DEDUPLICATION
# ============================================================================

class MetricDeduplicator:
    """
    ✅ FIX 16 + FIX 18: Enhanced deduplication.
    Group by normalized_metric, pick the one with highest confidence.
    Table override is handled earlier (paragraph values skipped if metric exists in table).
    """
    
    @classmethod
    def deduplicate(cls, metrics: List[Dict]) -> List[Dict]:
        """Return one best entry per normalized_metric."""
        if not metrics:
            return []

        groups: Dict[str, List[Dict]] = defaultdict(list)
        for m in metrics:
            groups[m['normalized_metric']].append(m)

        winners = []
        for metric_name, group in groups.items():
            winners.append(cls._pick_winner(group))

        return winners

    @classmethod
    def _pick_winner(cls, group: List[Dict]) -> Dict:
        """Pick the single best entry (confidence DESC, value DESC)."""
        return max(group, key=lambda m: (m['confidence'], m['value']))


# ============================================================================
# COMPLETE PIPELINE (with all new rules)
# ============================================================================

class PDFEvaluationPipeline:
    """
    ✅ FIX 33: TABLE OVERRIDE — If metric exists in table, ignore paragraph values.
    ✅ FIX 34-39: All new rules integrated.
    """
    
    def __init__(
        self,
        esg_filter_path: Optional[str] = None,
        ner_model_path: str = './models/ner_model/final',
        classifier_path: str = './models/classifier/final'
    ):
        print("\n" + "="*70)
        print("INITIALIZING ESG EXTRACTION PIPELINE v8.0 (TOTAL-ONLY ESG INTELLIGENCE)")
        print("="*70)
        print("\n🔒 APPLIED FIXES (v8.0):")
        print("  ✅ FIX 33: TABLE OVERRIDE — If metric exists in table, ignore paragraph values")
        print("  ✅ FIX 34: TOTAL ROW DETECTION — Only accept rows containing total/grand total/overall or last row")
        print("  ✅ FIX 35: VALUE SCALE VALIDATION — Minimum thresholds (Scope1/2≥1000, Scope3≥10000, Water≥100k, Energy≥1e8 MJ)")
        print("  ✅ FIX 36: STRICT UNIT VALIDATION — Waste→tonnes/kg, Emissions→tCO2e, Energy→MJ/kWh, Water→KL/m3")
        print("  ✅ FIX 37: PARAGRAPH FILTER — Reject if contains intervention/saved/reduced/initiative")
        print("  ✅ FIX 38: ESG SCORE HARD FILTER — Only if explicitly 'ESG score' or 'rating'")
        print("  ✅ FIX 39: CONFIDENCE CORRECTION — Table+total=0.95, table only=0.75, paragraph max 0.6")
        print()

        self.pdf_extractor    = PDFExtractor()
        self.table_parser     = EnhancedTableParser()  # primary
        self.preprocessor     = TextPreprocessor()
        self.esg_filter       = ESGCandidateFilter(esg_filter_path)
        self.ner_extractor    = ESGNERExtractor(ner_model_path)
        self.classifier       = MetricClassifier(classifier_path)
        self.value_extractor  = ValueExtractor()
        self.validator        = MetricValidator()
        self.confidence_scorer = ConfidenceScorer()
        self.deduplicator     = MetricDeduplicator()

        print("✓ Pipeline v8.0 (TOTAL-ONLY) initialized successfully\n")

    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a PDF and extract ESG metrics — tables first (total rows only), text fallback with strict filters."""
        print("\n" + "="*70)
        print(f"PROCESSING: {pdf_path}")
        print("="*70)

        results = {
            'file':             str(pdf_path),
            'pipeline_version': 'v8.0_total_only',
            'metrics':          [],
            'discarded':        [],
            'statistics':       {},
            'warnings':         [],
            'discard_reasons':  defaultdict(int),
        }

        # Layer 1: Extract text and tables
        print("\n[Layer 1] Extracting text and tables from PDF...")
        pdf_data = self.pdf_extractor.extract_text_and_tables(pdf_path)

        # Layer 1b: Enhanced table parser (primary)
        print("[Layer 1b] Running enhanced table parser (total-row only)...")
        table_metrics = EnhancedTableParser.parse_tables(
            pdf_data.get('tables', []),
            page_texts=pdf_data.get('pages', [])  # for fallback
        )
        print(f"  → {len(table_metrics)} table row(s) extracted")

        # Track which metrics were found in tables (for table override)
        table_metric_names: Set[str] = {m['normalized_metric'] for m in table_metrics}

        # Layer 2-3: Preprocess and chunk text (fallback)
        print("[Layer 2-3] Preprocessing and chunking text...")
        clean_text = self.preprocessor.clean_text(pdf_data['full_text'])
        chunks = self.preprocessor.chunk_text(clean_text)
        print(f"  → Created {len(chunks)} text chunks")

        # Layer 4: ESG filter
        print("[Layer 4] Filtering ESG candidates...")
        esg_chunks = []
        for chunk in chunks:
            is_esg, conf = self.esg_filter.is_esg_candidate(chunk['text'])
            if is_esg:
                chunk['esg_confidence'] = conf
                esg_chunks.append(chunk)

        esg_pct = len(esg_chunks) / len(chunks) * 100 if chunks else 0
        print(f"  → {len(esg_chunks)}/{len(chunks)} ESG chunks ({esg_pct:.1f}%)")

        # Layer 5-11: NER text path (secondary, only for metrics not in tables)
        text_metrics: List[Dict] = []
        all_discarded: List[Dict] = []

        for chunk_idx, chunk in enumerate(esg_chunks):
            # ✅ FIX 37: Paragraph filter before NER
            if PARAGRAPH_REJECT_KEYWORDS.search(chunk['text']):
                all_discarded.append({
                    'entity_text': chunk['text'][:100],
                    'reason': 'paragraph_reject_keyword',
                    'chunk_idx': chunk_idx,
                })
                results['discard_reasons']['paragraph_reject_keyword'] += 1
                continue

            entities = self.ner_extractor.extract_entities(
                chunk['text'], chunk['section_type']
            )
            if not entities:
                continue

            for entity in entities:
                metric_result, discard_reason = self._process_entity(entity, chunk)

                if discard_reason:
                    results['discard_reasons'][discard_reason] += 1
                    all_discarded.append({
                        'entity_text': entity['text'],
                        'reason':      discard_reason,
                        'chunk_idx':   chunk_idx,
                    })
                elif metric_result:
                    # ✅ FIX 33: TABLE OVERRIDE — ignore paragraph values if metric already found in table
                    if metric_result['normalized_metric'] not in table_metric_names:
                        text_metrics.append(metric_result)

        print(f"\n[Pre-dedup] Table: {len(table_metrics)}, Text: {len(text_metrics)}, "
              f"Discarded: {len(all_discarded)}")

        if results['discard_reasons']:
            print("\n[Discard Breakdown]")
            for reason, count in sorted(
                results['discard_reasons'].items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason}: {count}")

        # Combine: table rows always take precedence over text rows
        all_metrics = table_metrics + text_metrics

        # Deduplication — one winner per metric
        results['metrics'] = self.deduplicator.deduplicate(all_metrics)
        results['metrics'].sort(key=lambda x: x['confidence'], reverse=True)
        results['discarded'] = all_discarded

        # Confidence gate
        before_conf = len(results['metrics'])
        results['metrics'] = [m for m in results['metrics'] if m['confidence'] >= 0.5]
        discarded_low_conf = before_conf - len(results['metrics'])
        if discarded_low_conf > 0:
            print(f"\n[Conf gate] Discarded {discarded_low_conf} metrics with confidence < 0.5")

        print(f"[Final] {len(results['metrics'])} unique metrics")

        # Statistics
        valid_metrics = [m for m in results['metrics'] if m['validation_status'] == 'VALID']
        confidences   = [m['confidence'] for m in results['metrics']]
        results['statistics'] = {
            'total_chunks':             len(chunks),
            'esg_chunks':               len(esg_chunks),
            'esg_chunk_percentage':     round(esg_pct, 2),
            'table_extractions':        len(table_metrics),
            'text_extractions':         len(text_metrics),
            'discarded_extractions':    len(all_discarded),
            'discarded_low_confidence': discarded_low_conf,
            'final_metrics':            len(results['metrics']),
            'valid_metrics':            len(valid_metrics),
            'avg_confidence':  round(float(np.mean(confidences)), 4) if confidences else 0,
            'min_confidence':  round(float(np.min(confidences)), 4) if confidences else 0,
            'max_confidence':  round(float(np.max(confidences)), 4) if confidences else 0,
        }

        for metric in results['metrics']:
            if metric['validation_status'] != 'VALID':
                results['warnings'].append({
                    'metric': metric['normalized_metric'],
                    'issues': metric['validation_issues'],
                })

        # ✅ FIX 25: Build master-prompt-compatible JSON output
        mp_json = self._build_master_prompt_json(results['metrics'])
        results['master_prompt_output'] = mp_json

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {output_path}")

            # Also write the clean master-prompt JSON separately
            mp_path = output_path.replace('.json', '_mp_output.json')
            with open(mp_path, 'w') as f:
                json.dump(mp_json, f, indent=2)
            print(f"✓ Master-prompt JSON saved to {mp_path}")

        self._print_summary(results)
        return results

    def _process_entity(self, entity: Dict, chunk: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Process a single NER entity through classification, value extraction, validation.
        Includes ✅ FIX 38: ESG_SCORE hard filter.
        """
        # Extract context
        context = self.value_extractor.extract_context(
            chunk['text'], entity['start'], entity['end']
        )

        # ✅ FIX 1 + FIX 2: Classify with confidence gating
        normalized_metric, class_conf = self.classifier.classify(
            entity['text'], context, entity['section_type']
        )

        # ✅ FIX 2: EARLY TERMINATION FOR UNKNOWN
        if normalized_metric == "UNKNOWN":
            return None, "unknown_metric_classification"

        # ✅ FIX 38: ESG_SCORE hard filter — explicit label required
        if normalized_metric == 'ESG_SCORE':
            entity_lower = entity['text'].lower()
            if not re.search(r'\b(esg score|esg rating|sustainability score|sustainability rating)\b', entity_lower):
                return None, "esg_score_not_explicit"

        # Locate entity within context
        entity_text_in_context = entity['text']
        entity_pos_in_context  = context.find(entity_text_in_context)

        if entity_pos_in_context >= 0:
            ent_start_ctx = entity_pos_in_context
            ent_end_ctx   = entity_pos_in_context + len(entity_text_in_context)
        else:
            ent_start_ctx = None
            ent_end_ctx   = None

        # ✅ FIX 9/10/11: Value extraction with window + table + fallback
        value_data = self.value_extractor.extract_value(
            context,
            entity_start_in_context=ent_start_ctx,
            entity_end_in_context=ent_end_ctx,
            classifier_confidence=class_conf,
        )

        # ✅ FIX 13: NO EARLY DISCARD — if still no value data after all stages, then discard
        if not value_data:
            return None, "no_valid_value_after_hard_filtering"

        # ✅ FIX 17: Canonical unit normalisation
        canonical_unit, canonical_value = ValueExtractor.canonicalize_unit(
            value_data['unit'], normalized_metric, value_data['value']
        )
        value_data['unit']  = canonical_unit
        value_data['value'] = canonical_value

        # ✅ FIX 35: Value scale validation (reject small values)
        min_val = MIN_VALUES.get(normalized_metric, 0)
        if canonical_value < min_val:
            return None, f"value_below_min_{min_val}"

        # ✅ FIX 36: Strict unit validation (double-check)
        if normalized_metric in STRICT_UNIT_MAP and STRICT_UNIT_MAP[normalized_metric]:
            if canonical_unit and canonical_unit != 'UNKNOWN' and canonical_unit not in STRICT_UNIT_MAP[normalized_metric]:
                return None, f"invalid_unit_{canonical_unit}_for_{normalized_metric}"

        source_type = 'paragraph'  # text extraction path is always paragraph

        # ✅ FIX 5 + FIX 38: Validate (ESG_SCORE strict; UNKNOWN unit allowed with penalty)
        validation_status, validation_issues, validation_penalty = self.validator.validate(
            metric=normalized_metric,
            value=canonical_value,
            unit=canonical_unit,
            context=context,
            entity_text=entity['text'],
        )

        # ✅ FIX 8: EARLY FILTERING - discard only if validation penalty is exactly 0
        if validation_penalty == 0.0:
            return None, f"validation_discard: {'; '.join(validation_issues)}"

        # Check keyword match
        keyword_match = True
        if normalized_metric in METRIC_KEYWORDS:
            combined = f"{entity['text']} {context}".lower()
            keyword_match = any(kw in combined for kw in METRIC_KEYWORDS[normalized_metric])

        # ✅ FIX 6 + FIX 39: Calculate confidence with corrected range and paragraph cap
        confidence = self.confidence_scorer.calculate(
            ner_confidence=entity['confidence'],
            classification_confidence=class_conf,
            value_confidence=value_data['confidence'],
            unit=canonical_unit,
            entity_text=entity['text'],
            num_value_candidates=value_data.get('num_candidates', 1),
            validation_penalty=validation_penalty,
            keyword_match=keyword_match,
            metric=normalized_metric,
            source_type=source_type,
        )

        # Build result
        metric_result = {
            'entity_text':        entity['text'],
            'normalized_metric':  normalized_metric,
            'value':              canonical_value,
            'unit':               canonical_unit,
            'context':            context[:200],
            'section_type':       entity['section_type'],
            'confidence':         confidence,
            'validation_status':  validation_status,
            'validation_issues':  validation_issues,
            'source_type':        source_type,
        }

        return metric_result, None

    @staticmethod
    def _confidence_band(conf: float) -> str:
        """✅ FIX 25: Part 8 confidence band label."""
        if conf >= 0.9:  return "high (0.9-1.0)"
        if conf >= 0.7:  return "good (0.7-0.8)"
        if conf >= 0.5:  return "weak (0.5-0.6)"
        return "discard (<0.5)"

    @staticmethod
    def _build_master_prompt_json(metrics: List[Dict]) -> Dict:
        """
        ✅ FIX 25: Produce the clean JSON format specified in master prompt Part 7.
        """
        output: Dict = {}
        for m in metrics:
            output[m['normalized_metric']] = {
                'value':            m['value'],
                'unit':             m.get('unit', ''),
                'confidence':       m['confidence'],
                'confidence_band':  PDFEvaluationPipeline._confidence_band(m['confidence']),
                'source':           m.get('source_type', 'text'),
                'page':             m.get('page', None),
            }
        return output
    
    def _print_summary(self, results: Dict):
        """Print summary of extraction results"""
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY (v8.0 TOTAL-ONLY ESG INTELLIGENCE)")
        print("="*70)
        
        stats = results['statistics']
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  ESG chunks: {stats['esg_chunks']} ({stats['esg_chunk_percentage']}%)")
        print(f"  Table extractions: {stats['table_extractions']}")
        print(f"  Text extractions: {stats['text_extractions']}")
        print(f"  Discarded: {stats['discarded_extractions']}")
        print(f"  Discarded (low confidence): {stats['discarded_low_confidence']}")
        print(f"  Final unique metrics: {stats['final_metrics']}")
        print(f"  Valid metrics: {stats['valid_metrics']}")
        
        print(f"\n📈 Confidence Statistics:")
        print(f"  Average: {stats['avg_confidence']:.4f}")
        print(f"  Min: {stats['min_confidence']:.4f}")
        print(f"  Max: {stats['max_confidence']:.4f}")
        
        print(f"\n✅ Target Achievements:")
        if 70 <= stats['esg_chunk_percentage'] <= 85:
            print(f"  ✓ ESG chunk rate in target range (70-85%): {stats['esg_chunk_percentage']:.1f}%")
        else:
            print(f"  ⚠ ESG chunk rate outside target: {stats['esg_chunk_percentage']:.1f}% (target: 70-85%)")
        
        if 0.55 <= stats['avg_confidence'] <= 0.75:
            print(f"  ✓ Avg confidence in target range (55-75%): {stats['avg_confidence']:.2%}")
        else:
            print(f"  ⚠ Avg confidence outside target: {stats['avg_confidence']:.2%} (target: 55-75%)")
        
        print(f"\n📋 Extracted Metrics:")
        if results['metrics']:
            metric_counts = defaultdict(int)
            for m in results['metrics']:
                metric_counts[m['normalized_metric']] += 1
            
            for metric, count in sorted(metric_counts.items()):
                print(f"  - {metric}: {count}")
            
            print(f"\n🔝 Top 5 Metrics (by confidence):")
            for i, m in enumerate(results['metrics'][:5], 1):
                print(f"  {i}. {m['normalized_metric']}: {m['value']} {m.get('unit', '')} "
                      f"(conf: {m['confidence']:.4f})")
        else:
            print("  (No metrics extracted)")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings: {len(results['warnings'])}")
            for w in results['warnings'][:5]:
                print(f"  - {w['metric']}: {', '.join(w['issues'][:2])}")
        
        print("\n" + "="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ESG PDF Extraction v8.0 (Total-Only)')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--output_path', type=str, default='results_v8.0_total_only.json',
                       help='Output JSON path')
    parser.add_argument('--ner_model', type=str, default='./models/ner_model/final',
                       help='NER model path')
    parser.add_argument('--classifier', type=str, default='./models/classifier/final',
                       help='Classifier model path')
    parser.add_argument('--esg_filter', type=str, default=None,
                       help='ESG filter model path (optional)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PDFEvaluationPipeline(
        esg_filter_path=args.esg_filter,
        ner_model_path=args.ner_model,
        classifier_path=args.classifier
    )
    
    # Process PDF
    results = pipeline.process_pdf(args.pdf_path, args.output_path)
    
    print(f"\n✅ Processing complete!")
    print(f"📄 Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()