"""
ESG METRIC EXTENSIONS — Plugin Module for Social/Governance Metrics
====================================================================

Adds extraction support for the 5 non-environmental target metrics:
  - GENDER_DIVERSITY
  - SAFETY_INCIDENTS
  - EMPLOYEE_WELLBEING
  - DATA_BREACHES
  - COMPLAINTS

This module is designed as a NON-INTRUSIVE plugin:
  - No existing extraction logic is modified
  - All functions are NEW and self-contained
  - Integration is via a single orchestrator: run_extended_extraction()

Usage:
    from metric_extensions import run_extended_extraction
    extended_metrics = run_extended_extraction(pdf_data, existing_metrics)
"""

import re
from typing import List, Dict, Optional, Set


# ============================================================================
# NEW METRIC NAMES
# ============================================================================

NEW_METRICS: Set[str] = {
    'GENDER_DIVERSITY',
    'SAFETY_INCIDENTS',
    'EMPLOYEE_WELLBEING',
    'DATA_BREACHES',
    'COMPLAINTS',
}


# ============================================================================
# PATTERNS FOR METRIC EXTRACTION
# ============================================================================

METRIC_PATTERNS: Dict[str, List[re.Pattern]] = {
    'GENDER_DIVERSITY': [
        re.compile(r'gender\s+diversity', re.I),
        re.compile(r'women\s+(?:in\s+)?(?:workforce|employees)', re.I),
        re.compile(r'female\s+(?:employees|representation|workforce)', re.I),
        re.compile(r'women\s+employees', re.I),
        re.compile(r'workforce\s+diversity', re.I),
        re.compile(r'gender\s+ratio', re.I),
    ],
    'SAFETY_INCIDENTS': [
        re.compile(r'safety\s+incidents?', re.I),
        re.compile(r'lost\s+time\s+injur', re.I),
        re.compile(r'\bltifr\b', re.I),
        re.compile(r'fatalit(?:y|ies)', re.I),
        re.compile(r'recordable\s+incidents?', re.I),
        re.compile(r'occupational\s+injur', re.I),
        re.compile(r'workplace\s+accidents?', re.I),
        re.compile(r'injury\s+(?:rate|frequency)', re.I),
    ],
    'EMPLOYEE_WELLBEING': [
        re.compile(r'employee\s+well.?being', re.I),
        re.compile(r'training\s+hours', re.I),
        re.compile(r'employee\s+turnover', re.I),
        re.compile(r'attrition\s+rate', re.I),
        re.compile(r'employee\s+satisfaction', re.I),
    ],
    'DATA_BREACHES': [
        re.compile(r'data\s+breach', re.I),
        re.compile(r'cyber\s+(?:security\s+)?incident', re.I),
        re.compile(r'privacy\s+breach', re.I),
        re.compile(r'security\s+incident', re.I),
        re.compile(r'data\s+leak', re.I),
    ],
    'COMPLAINTS': [
        re.compile(r'complaints?\s+(?:received|filed|reported)', re.I),
        re.compile(r'grievance', re.I),
        re.compile(r'whistleblower\s+complaint', re.I),
        re.compile(r'ethics\s+(?:complaint|violation)', re.I),
        re.compile(r'consumer\s+complaints?', re.I),
        re.compile(r'customer\s+complaints?', re.I),
        re.compile(r'number\s+of\s+complaints?', re.I),
    ],
}


# ============================================================================
# REJECT PATTERNS (reused from TableReconstructor concept)
# ============================================================================

_REJECT_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bper\s+(employee|fte|unit|tonne|kwh|mwh|revenue|capita)', re.I),
    re.compile(r'\bintensity\b', re.I),
    re.compile(r'\btarget\b', re.I),
    re.compile(r'\breduction\b', re.I),
    re.compile(r'\bprojection\b', re.I),
    re.compile(r'\bforecast\b', re.I),
    re.compile(r'\bbaseline\b', re.I),
    re.compile(r'\bgoal\b', re.I),
]


def _is_rejected(text: str) -> bool:
    """Check if row contains intensity/target/reduction patterns."""
    for pat in _REJECT_PATTERNS:
        if pat.search(text):
            return True
    return False


def _extract_first_valid_number(text: str, min_value: float = 0.0) -> Optional[float]:
    """
    Extract the FIRST valid number from text.
    Skip noise values (page numbers, years, IDs).
    """
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    for num_str in numbers:
        try:
            value = float(num_str.replace(",", ""))
        except ValueError:
            continue

        # Noise filtering
        if value < min_value:
            continue
        if 1900 <= value <= 2100:  # Year
            continue
        if value in {1, 2, 3, 4, 5, 10}:  # Noise
            continue

        return value
    return None


# ============================================================================
# METRIC EXTRACTION — MAIN HELPER
# ============================================================================

def extract_metrics_from_tables(all_rows: List[Dict]) -> List[Dict]:
    """
    Extract social/governance metrics from table rows.

    Rules:
      - Detect rows containing metric-related patterns
      - Extract FIRST valid numeric value
      - Apply metric-specific validation
      - Ignore intensity/target rows

    Args:
        all_rows: List of dicts with 'text' and 'page' keys (normalized table rows)

    Returns:
        List of metric dicts for found metrics
    """
    results = []

    for metric_name, patterns in METRIC_PATTERNS.items():
        found = False
        for pattern in patterns:
            if found:
                break
            for row_info in all_rows:
                row_text = row_info['text']

                if _is_rejected(row_text):
                    continue

                if pattern.search(row_text):
                    value = _extract_first_valid_number(row_text, min_value=0.0)

                    if value is not None and _validate_metric_value(metric_name, value):
                        # Determine confidence
                        has_total = 'total' in row_text.lower() or 'overall' in row_text.lower()
                        confidence = 0.90 if has_total else 0.80

                        unit = _get_unit_for_metric(metric_name)

                        results.append({
                            'normalized_metric': metric_name,
                            'value': value,
                            'unit': unit,
                            'entity_text': row_text[:100],
                            'context': row_text[:200],
                            'section_type': _get_section_type(metric_name),
                            'confidence': confidence,
                            'validation_status': 'VALID',
                            'validation_issues': [],
                            'source_type': 'table_reconstructed',
                            'page': row_info['page'],
                        })
                        print(f"    ✅ [EXT] {metric_name}: {value} {unit} "
                              f"(page {row_info['page']})")
                        found = True
                        break

    return results


# ============================================================================
# VALIDATION FOR NEW METRICS
# ============================================================================

def _validate_metric_value(metric_name: str, value: float) -> bool:
    """Validate a metric value based on expected ranges."""
    ranges = {
        'GENDER_DIVERSITY': (0, 100),       # percentage
        'SAFETY_INCIDENTS': (0, 100_000),   # count
        'EMPLOYEE_WELLBEING': (0, 100),     # percentage or score
        'DATA_BREACHES': (0, 100_000),      # count
        'COMPLAINTS': (0, 1_000_000),       # count
    }
    if metric_name in ranges:
        min_val, max_val = ranges[metric_name]
        return min_val <= value <= max_val
    return True


def validate_new_metric(metric: Dict) -> bool:
    """
    Validate a new metric result.

    Args:
        metric: Metric dict to validate

    Returns:
        True if valid, False if should be discarded
    """
    name = metric['normalized_metric']
    value = metric['value']

    if not _validate_metric_value(name, value):
        print(f"    ❌ [EXT-VALIDATE] {name}: value {value} outside valid range")
        metric['validation_status'] = 'INVALID'
        metric['validation_issues'].append(f"Value {value} outside valid range for {name}")
        return False

    return True


# ============================================================================
# HELPER: Unit and section type for metrics
# ============================================================================

def _get_unit_for_metric(metric_name: str) -> str:
    """Return the expected unit for a metric."""
    unit_map = {
        'GENDER_DIVERSITY': '%',
        'SAFETY_INCIDENTS': '',    # count-based
        'EMPLOYEE_WELLBEING': '%',
        'DATA_BREACHES': '',       # count-based
        'COMPLAINTS': '',          # count-based
    }
    return unit_map.get(metric_name, '')


def _get_section_type(metric_name: str) -> str:
    """Return the ESG section type for a metric."""
    section_map = {
        'GENDER_DIVERSITY': 'Social',
        'SAFETY_INCIDENTS': 'Social',
        'EMPLOYEE_WELLBEING': 'Social',
        'DATA_BREACHES': 'Governance',
        'COMPLAINTS': 'Governance',
    }
    return section_map.get(metric_name, 'Unknown')


# ============================================================================
# HELPER: Collect normalized rows from PDF data
# ============================================================================

def _collect_normalized_rows(pdf_data: Dict) -> List[Dict]:
    """
    Collect normalized table rows from PDF data.
    Replicates TableReconstructor's row collection logic without importing it.
    """
    all_rows = []

    # From grid tables
    for table_info in pdf_data.get('tables', []):
        raw_table = table_info['table']
        page_num = table_info['page']

        for row in raw_table:
            if not row:
                continue
            row_text = " ".join(
                [str(cell).strip() for cell in row if cell and str(cell).strip()]
            )
            if row_text:
                all_rows.append({'text': row_text, 'page': page_num})

    # From page text (text-based tables)
    for page_info in pdf_data.get('pages', []):
        text = page_info.get('text', '')
        page_num = page_info.get('page_number', 0)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.search(r'[\d,]+\.?\d*', line):
                all_rows.append({'text': line, 'page': page_num})

    return all_rows


# ============================================================================
# ORCHESTRATOR — MAIN ENTRY POINT
# ============================================================================

def run_extended_extraction(pdf_data: Dict, existing_metrics: List[Dict]) -> List[Dict]:
    """
    Run extended metric extraction as a plugin.

    This is the SINGLE entry point for the pipeline to call.
    It runs AFTER all existing extraction stages and BEFORE deduplication.

    Pipeline:
      1. Collect normalized rows from PDF data
      2. Extract social/governance metrics from tables
      3. Validate all new metrics
      4. Return only valid new metrics (not already in existing_metrics)

    Args:
        pdf_data: Raw PDF data dict from PDFExtractor
        existing_metrics: All metrics found by existing pipeline stages

    Returns:
        List of new metric dicts to append to existing results
    """
    print("\n" + "-" * 50)
    print("[STAGE 4] Extended metric extraction (social/governance)...")
    print("-" * 50)

    # Step 1: Collect rows
    all_rows = _collect_normalized_rows(pdf_data)
    print(f"    [EXT] {len(all_rows)} rows collected for extended extraction")

    # Track what's already found
    existing_names = {m['normalized_metric'] for m in existing_metrics}

    extended_results: List[Dict] = []

    # Step 2: Extract social/governance metrics
    metrics = extract_metrics_from_tables(all_rows)
    for metric in metrics:
        if metric['normalized_metric'] not in existing_names:
            extended_results.append(metric)
            existing_names.add(metric['normalized_metric'])

    # Step 3: Validate
    valid_results = []
    for metric in extended_results:
        if validate_new_metric(metric):
            valid_results.append(metric)

    print(f"    → Stage 4: {len(valid_results)} new metric(s) extracted via extension plugin")

    for m in valid_results:
        print(f"      • {m['normalized_metric']}: {m['value']} {m.get('unit', '')} "
              f"(conf: {m['confidence']}, src: {m['source_type']})")

    return valid_results
