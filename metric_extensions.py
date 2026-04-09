"""
ESG METRIC EXTENSIONS — Plugin Module for 4 New Metrics
========================================================

Adds extraction support for:
  - ENVIRONMENTAL_SCORE
  - SOCIAL_SCORE
  - GOVERNANCE_SCORE
  - CARBON_EMISSIONS

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
    'ENVIRONMENTAL_SCORE',
    'SOCIAL_SCORE',
    'GOVERNANCE_SCORE',
    'CARBON_EMISSIONS',
}

SCORE_METRICS: Set[str] = {
    'ENVIRONMENTAL_SCORE',
    'SOCIAL_SCORE',
    'GOVERNANCE_SCORE',
}


# ============================================================================
# PATTERNS FOR SCORE EXTRACTION
# ============================================================================

SCORE_PATTERNS: Dict[str, List[re.Pattern]] = {
    'ENVIRONMENTAL_SCORE': [
        re.compile(r'environmental\s+score', re.I),
        re.compile(r'environment\s+score', re.I),
        re.compile(r'\bE\s+score\b', re.I),
        re.compile(r'environmental\s+pillar\s+score', re.I),
        re.compile(r'environmental\s+rating', re.I),
        re.compile(r'environment\s+rating', re.I),
        re.compile(r'environmental\s+performance\s+score', re.I),
    ],
    'SOCIAL_SCORE': [
        re.compile(r'social\s+score', re.I),
        re.compile(r'\bS\s+score\b', re.I),
        re.compile(r'social\s+pillar\s+score', re.I),
        re.compile(r'social\s+rating', re.I),
        re.compile(r'social\s+performance\s+score', re.I),
    ],
    'GOVERNANCE_SCORE': [
        re.compile(r'governance\s+score', re.I),
        re.compile(r'\bG\s+score\b', re.I),
        re.compile(r'governance\s+pillar\s+score', re.I),
        re.compile(r'governance\s+rating', re.I),
        re.compile(r'governance\s+performance\s+score', re.I),
        re.compile(r'corporate\s+governance\s+score', re.I),
    ],
}


# ============================================================================
# PATTERNS FOR CARBON EMISSIONS EXTRACTION
# ============================================================================

CARBON_PATTERNS: List[re.Pattern] = [
    re.compile(r'total\s+(?:carbon\s+)?emissions', re.I),
    re.compile(r'total\s+ghg\s+emissions', re.I),
    re.compile(r'total\s+greenhouse\s+gas\s+emissions', re.I),
    re.compile(r'overall\s+(?:carbon\s+)?emissions', re.I),
    re.compile(r'overall\s+ghg\s+emissions', re.I),
    re.compile(r'total\s+co2\s+emissions', re.I),
    re.compile(r'total\s+co2e\s+emissions', re.I),
    re.compile(r'aggregate\s+(?:carbon\s+)?emissions', re.I),
    re.compile(r'combined\s+scope\s+emissions', re.I),
    re.compile(r'total\s+scope\s+1.*2.*3\s+emissions', re.I),
]


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


def _extract_first_valid_number(text: str, min_value: float = 1.0) -> Optional[float]:
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


def _detect_emission_unit(text: str) -> str:
    """Detect emission unit from row text."""
    text_lower = text.lower()
    if 'mtco2' in text_lower or 'mt co2' in text_lower:
        return 'MtCO2e'
    if 'ktco2' in text_lower:
        return 'ktCO2e'
    if 'tco2' in text_lower or 'tonnes co2' in text_lower or 'metric tonnes' in text_lower:
        return 'tCO2e'
    if 'co2' in text_lower or 'carbon' in text_lower or 'ghg' in text_lower:
        return 'tCO2e'
    if 'tonnes' in text_lower or 'metric ton' in text_lower:
        return 'tCO2e'
    return 'tCO2e'  # default for emissions


# ============================================================================
# SCORE EXTRACTION — NEW HELPER
# ============================================================================

def extract_scores_from_tables(all_rows: List[Dict]) -> List[Dict]:
    """
    Extract Environmental, Social, and Governance scores from table rows.

    Rules:
      - Detect rows containing score-related patterns
      - Extract FIRST valid numeric value
      - Value must be 0–100 (score range)
      - Ignore intensity/target rows
      - Unit is empty (scores are unitless)

    Args:
        all_rows: List of dicts with 'text' and 'page' keys (normalized table rows)

    Returns:
        List of metric dicts for found scores
    """
    results = []

    for metric_name, patterns in SCORE_PATTERNS.items():
        found = False
        for pattern in patterns:
            if found:
                break
            for row_info in all_rows:
                row_text = row_info['text']

                if _is_rejected(row_text):
                    continue

                if pattern.search(row_text):
                    value = _extract_first_valid_number(row_text, min_value=0.1)

                    if value is not None and 0 <= value <= 100:
                        # Determine confidence
                        has_total = 'total' in row_text.lower() or 'overall' in row_text.lower()
                        confidence = 0.90 if has_total else 0.85

                        results.append({
                            'normalized_metric': metric_name,
                            'value': value,
                            'unit': '',  # scores are unitless
                            'entity_text': row_text[:100],
                            'context': row_text[:200],
                            'section_type': _get_section_type(metric_name),
                            'confidence': confidence,
                            'validation_status': 'VALID',
                            'validation_issues': [],
                            'source_type': 'table_reconstructed',
                            'page': row_info['page'],
                        })
                        print(f"    ✅ [EXT] {metric_name}: {value} "
                              f"(page {row_info['page']})")
                        found = True
                        break

    return results


# ============================================================================
# CARBON EMISSIONS EXTRACTION — NEW HELPER
# ============================================================================

def extract_carbon_from_tables(all_rows: List[Dict]) -> Optional[Dict]:
    """
    Extract total carbon emissions from table rows.

    Rules:
      - Detect rows with "total emissions", "total carbon emissions",
        "total ghg emissions", "overall emissions"
      - Extract first valid numeric value >= 100
      - Assign unit tCO2e (default if missing)
      - Ignore intensity/target rows

    Args:
        all_rows: List of dicts with 'text' and 'page' keys

    Returns:
        Metric dict if found, else None
    """
    for pattern in CARBON_PATTERNS:
        for row_info in all_rows:
            row_text = row_info['text']

            if _is_rejected(row_text):
                continue

            # Avoid matching individual scope rows
            row_lower = row_text.lower()
            if re.search(r'scope\s*[123]\b', row_lower) and 'total' not in row_lower:
                continue

            if pattern.search(row_text):
                value = _extract_first_valid_number(row_text, min_value=100)

                if value is not None and value >= 100:
                    unit = _detect_emission_unit(row_text)
                    has_total = 'total' in row_lower or 'overall' in row_lower
                    confidence = 0.90 if has_total else 0.70

                    result = {
                        'normalized_metric': 'CARBON_EMISSIONS',
                        'value': value,
                        'unit': unit,
                        'entity_text': row_text[:100],
                        'context': row_text[:200],
                        'section_type': 'Environmental',
                        'confidence': confidence,
                        'validation_status': 'VALID',
                        'validation_issues': [],
                        'source_type': 'table_reconstructed',
                        'page': row_info['page'],
                    }
                    print(f"    ✅ [EXT] CARBON_EMISSIONS: {value} {unit} "
                          f"(page {row_info['page']})")
                    return result

    return None


# ============================================================================
# DERIVED CARBON EMISSIONS (Scope 1 + 2 + 3)
# ============================================================================

def derive_carbon_emissions(existing_metrics: List[Dict]) -> Optional[Dict]:
    """
    Compute Carbon Emissions = Scope 1 + Scope 2 + Scope 3.

    Only runs if:
      - CARBON_EMISSIONS is NOT already found
      - All three scopes (SCOPE_1, SCOPE_2, SCOPE_3) exist

    Args:
        existing_metrics: List of all metric dicts found so far

    Returns:
        Derived CARBON_EMISSIONS metric dict, or None
    """
    # Check if carbon already exists
    metric_names = {m['normalized_metric'] for m in existing_metrics}
    if 'CARBON_EMISSIONS' in metric_names:
        return None

    # Collect scope values
    scope_values = {}
    scope_units = {}
    for m in existing_metrics:
        if m['normalized_metric'] in ('SCOPE_1', 'SCOPE_2', 'SCOPE_3'):
            scope_values[m['normalized_metric']] = m['value']
            scope_units[m['normalized_metric']] = m.get('unit', 'tCO2e')

    # Need all three
    if len(scope_values) < 3:
        return None

    total = sum(scope_values.values())

    # Use the most common unit among scopes (default tCO2e)
    unit_counts = {}
    for u in scope_units.values():
        unit_counts[u] = unit_counts.get(u, 0) + 1
    unit = max(unit_counts, key=unit_counts.get) if unit_counts else 'tCO2e'

    result = {
        'normalized_metric': 'CARBON_EMISSIONS',
        'value': round(total, 2),
        'unit': unit,
        'entity_text': f"Derived: Scope1({scope_values['SCOPE_1']}) + "
                       f"Scope2({scope_values['SCOPE_2']}) + "
                       f"Scope3({scope_values['SCOPE_3']})",
        'context': 'Computed as sum of Scope 1 + Scope 2 + Scope 3 emissions',
        'section_type': 'Environmental',
        'confidence': 0.85,
        'validation_status': 'VALID',
        'validation_issues': ['derived_from_scopes'],
        'source_type': 'derived',
        'page': 0,
    }

    print(f"    ✅ [EXT] CARBON_EMISSIONS (derived): {total} {unit} "
          f"(Scope1 + Scope2 + Scope3)")
    return result


# ============================================================================
# VALIDATION FOR NEW METRICS
# ============================================================================

def validate_new_metric(metric: Dict) -> bool:
    """
    Validate a new metric result.

    Rules:
      - Score metrics: value must be 0–100
      - Carbon emissions: must have valid emission unit

    Args:
        metric: Metric dict to validate

    Returns:
        True if valid, False if should be discarded
    """
    name = metric['normalized_metric']
    value = metric['value']
    unit = metric.get('unit', '')

    # Score validation: 0–100
    if name in SCORE_METRICS:
        if value < 0 or value > 100:
            print(f"    ❌ [EXT-VALIDATE] {name}: value {value} outside 0–100 range")
            metric['validation_status'] = 'INVALID'
            metric['validation_issues'].append(f"Score value {value} outside range [0, 100]")
            return False
        return True

    # Carbon emissions validation
    if name == 'CARBON_EMISSIONS':
        valid_carbon_units = {
            'tCO2e', 'Mt CO2e', 'MtCO2e', 'ktCO2e', 'tCO2',
            'tonnes CO2', 'tonnes', 't', 'MT'
        }
        if unit and unit not in valid_carbon_units:
            print(f"    ❌ [EXT-VALIDATE] CARBON_EMISSIONS: invalid unit '{unit}'")
            metric['validation_status'] = 'INVALID'
            metric['validation_issues'].append(f"Invalid unit '{unit}' for CARBON_EMISSIONS")
            return False
        if value < 100:
            print(f"    ❌ [EXT-VALIDATE] CARBON_EMISSIONS: value {value} too small")
            metric['validation_status'] = 'INVALID'
            metric['validation_issues'].append(f"Carbon value {value} below minimum 100")
            return False
        return True

    return True


# ============================================================================
# HELPER: Section type for scores
# ============================================================================

def _get_section_type(metric_name: str) -> str:
    """Return the ESG section type for a metric."""
    section_map = {
        'ENVIRONMENTAL_SCORE': 'Environmental',
        'SOCIAL_SCORE': 'Social',
        'GOVERNANCE_SCORE': 'Governance',
        'CARBON_EMISSIONS': 'Environmental',
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
      2. Extract scores (Environmental, Social, Governance)
      3. Extract carbon emissions from tables
      4. If carbon not found, derive from scopes
      5. Validate all new metrics
      6. Return only valid new metrics (not already in existing_metrics)

    Args:
        pdf_data: Raw PDF data dict from PDFExtractor
        existing_metrics: All metrics found by existing pipeline stages

    Returns:
        List of new metric dicts to append to existing results
    """
    print("\n" + "-" * 50)
    print("[STAGE 4] Extended metric extraction (plugin)...")
    print("-" * 50)

    # Step 1: Collect rows
    all_rows = _collect_normalized_rows(pdf_data)
    print(f"    [EXT] {len(all_rows)} rows collected for extended extraction")

    # Track what's already found
    existing_names = {m['normalized_metric'] for m in existing_metrics}

    extended_results: List[Dict] = []

    # Step 2: Extract scores
    scores = extract_scores_from_tables(all_rows)
    for score in scores:
        if score['normalized_metric'] not in existing_names:
            extended_results.append(score)
            existing_names.add(score['normalized_metric'])

    # Step 3: Extract carbon emissions from tables
    carbon = extract_carbon_from_tables(all_rows)
    if carbon and 'CARBON_EMISSIONS' not in existing_names:
        extended_results.append(carbon)
        existing_names.add('CARBON_EMISSIONS')

    # Step 4: Derive carbon if not found
    if 'CARBON_EMISSIONS' not in existing_names:
        # Combine existing + extended for derivation input
        all_for_derivation = existing_metrics + extended_results
        derived = derive_carbon_emissions(all_for_derivation)
        if derived:
            extended_results.append(derived)

    # Step 5: Validate
    valid_results = []
    for metric in extended_results:
        if validate_new_metric(metric):
            valid_results.append(metric)

    print(f"    → Stage 4: {len(valid_results)} new metric(s) extracted via extension plugin")

    for m in valid_results:
        print(f"      • {m['normalized_metric']}: {m['value']} {m.get('unit', '')} "
              f"(conf: {m['confidence']}, src: {m['source_type']})")

    return valid_results
