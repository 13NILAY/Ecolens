"""
Quick test: Run just the TableReconstructor on a PDF to verify structured extraction works.
This avoids the torch dependency needed for NER/classifier models.
"""
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

# PDF extraction
import pdfplumber

# ============================================================================
# CONSTANTS (copied from evaluate_on_pdf.py for standalone test)
# ============================================================================
CONFIDENCE_TABLE_TOTAL = 0.90
CONFIDENCE_TABLE_ONLY = 0.70

# ============================================================================
# PDF EXTRACTOR (standalone copy)
# ============================================================================
def extract_text_and_tables(pdf_path: str) -> Dict:
    result = {'full_text': '', 'pages': [], 'tables': []}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                result['pages'].append({'page_number': page_num, 'text': text})
                result['full_text'] += text + '\n\n'
            tables = page.extract_tables()
            if tables:
                for table_idx, table in enumerate(tables):
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [
                            cell.strip() if cell and isinstance(cell, str) else (cell if cell else '')
                            for cell in row
                        ]
                        if any(cell for cell in cleaned_row):
                            cleaned_table.append(cleaned_row)
                    if len(cleaned_table) > 1:
                        result['tables'].append({
                            'page': page_num,
                            'table': cleaned_table,
                            'table_index': table_idx
                        })
    return result

# ============================================================================
# TABLE RECONSTRUCTOR (copied from evaluate_on_pdf.py)
# ============================================================================
class TableReconstructor:
    SCOPE_PATTERNS = {
        'SCOPE_1': [
            re.compile(r'total\s+scope\s*1', re.I),
            re.compile(r'scope\s*1\s+emission', re.I),
            re.compile(r'scope\s*-?\s*1', re.I),
            re.compile(r'direct\s+emission', re.I),
        ],
        'SCOPE_2': [
            re.compile(r'total\s+scope\s*2', re.I),
            re.compile(r'scope\s*2\s+emission', re.I),
            re.compile(r'scope\s*-?\s*2', re.I),
            re.compile(r'indirect\s+emission', re.I),
        ],
        'SCOPE_3': [
            re.compile(r'total\s+scope\s*3', re.I),
            re.compile(r'scope\s*3\s+emission', re.I),
            re.compile(r'scope\s*-?\s*3', re.I),
            re.compile(r'other\s+indirect\s+emission', re.I),
            re.compile(r'value\s+chain\s+emission', re.I),
        ],
    }
    
    ENERGY_PATTERNS = [
        re.compile(r'total\s+electricity\s+consumption', re.I),
        re.compile(r'total\s+energy\s+consumption', re.I),
        re.compile(r'total\s+energy', re.I),
        re.compile(r'electricity\s+consumption', re.I),
        re.compile(r'energy\s+consumption', re.I),
    ]
    
    WASTE_PATTERNS = [
        re.compile(r'total\s+waste', re.I),
        re.compile(r'waste\s+generated', re.I),
        re.compile(r'hazardous\s+waste', re.I),
        re.compile(r'non.?hazardous\s+waste', re.I),
        re.compile(r'other\s+waste', re.I),
        re.compile(r'plastic\s+waste', re.I),
        re.compile(r'e.?waste', re.I),
        re.compile(r'bio.?medical\s+waste', re.I),
        re.compile(r'construction.*waste', re.I),
        re.compile(r'battery\s+waste', re.I),
    ]
    
    WATER_PATTERNS = [
        re.compile(r'total\s+water', re.I),
        re.compile(r'water\s+withdrawal', re.I),
        re.compile(r'water\s+consumption', re.I),
        re.compile(r'groundwater', re.I),
        re.compile(r'ground\s+water', re.I),
        re.compile(r'surface\s+water', re.I),
        re.compile(r'third.?party\s+water', re.I),
        re.compile(r'municipal\s+water', re.I),
        re.compile(r'tanker\s+water', re.I),
        re.compile(r'rainwater', re.I),
    ]
    
    REJECT_PATTERNS = [
        re.compile(r'\bper\s+(employee|fte|unit|tonne|kwh|mwh|revenue|capita)', re.I),
        re.compile(r'\bintensity\b', re.I),
        re.compile(r'\btarget\b', re.I),
        re.compile(r'\breduction\b', re.I),
        re.compile(r'\bprojection\b', re.I),
        re.compile(r'\bforecast\b', re.I),
        re.compile(r'\bbaseline\b', re.I),
        re.compile(r'\bgoal\b', re.I),
    ]
    
    @classmethod
    def extract_from_pdf(cls, pdf_data: Dict) -> List[Dict]:
        all_rows = []
        for table_info in pdf_data.get('tables', []):
            raw_table = table_info['table']
            page_num = table_info['page']
            normalized = cls._normalize_table(raw_table)
            for row_str in normalized:
                all_rows.append({'text': row_str, 'page': page_num})
        for page_info in pdf_data.get('pages', []):
            text_rows = cls._extract_text_table_rows(page_info['text'])
            for row_str in text_rows:
                all_rows.append({'text': row_str, 'page': page_info['page_number']})
        
        print(f"    [TableReconstructor] {len(all_rows)} normalized rows collected")
        
        metrics = {}
        for scope_metric, patterns in cls.SCOPE_PATTERNS.items():
            result = cls._extract_scope(all_rows, patterns, scope_metric)
            if result:
                metrics[scope_metric] = result
                print(f"    ✅ [TR] {scope_metric}: {result['value']} {result['unit']} (page {result['page']})")
        
        energy = cls._extract_energy(all_rows)
        if energy:
            metrics['ENERGY_CONSUMPTION'] = energy
            print(f"    ✅ [TR] ENERGY_CONSUMPTION: {energy['value']} {energy['unit']} (page {energy['page']})")
        
        waste = cls._extract_waste(all_rows)
        if waste:
            metrics['WASTE_GENERATED'] = waste
            print(f"    ✅ [TR] WASTE_GENERATED: {waste['value']} {waste['unit']} (page {waste['page']})")
        
        water = cls._extract_water(all_rows)
        if water:
            metrics['WATER_USAGE'] = water
            print(f"    ✅ [TR] WATER_USAGE: {water['value']} {water['unit']} (page {water['page']})")
        
        return list(metrics.values())
    
    @classmethod
    def _normalize_table(cls, table):
        raw_strings = []
        for row in table:
            if not row:
                continue
            row_text = " ".join([str(cell).strip() for cell in row if cell and str(cell).strip()])
            if row_text:
                raw_strings.append(row_text)
        
        normalized = []
        buffer = ""
        for row_text in raw_strings:
            if buffer:
                row_text = buffer + " " + row_text
                buffer = ""
            words = row_text.split()
            has_number = bool(re.search(r'\d', row_text))
            if len(words) < 3 and not has_number:
                buffer = row_text
            else:
                normalized.append(row_text)
        if buffer:
            if normalized:
                normalized[-1] = normalized[-1] + " " + buffer
            else:
                normalized.append(buffer)
        return normalized
    
    @classmethod
    def _extract_text_table_rows(cls, page_text):
        rows = []
        for line in page_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.search(r'[\d,]+\.?\d*', line):
                rows.append(line)
        return rows
    
    @classmethod
    def _is_rejected(cls, text):
        for pat in cls.REJECT_PATTERNS:
            if pat.search(text):
                return True
        return False
    
    @classmethod
    def _extract_first_valid_number(cls, text, min_value=1.0):
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        for num_str in numbers:
            try:
                value = float(num_str.replace(",", ""))
            except ValueError:
                continue
            if value < min_value:
                continue
            if 1900 <= value <= 2100:
                continue
            if value in {1, 2, 3, 4, 5, 10}:
                continue
            return value
        return None
    
    @classmethod
    def _extract_scope(cls, rows, patterns, metric_name):
        for pattern in patterns:
            for row_info in rows:
                row_text = row_info['text']
                if cls._is_rejected(row_text):
                    continue
                if pattern.search(row_text):
                    value = cls._extract_first_valid_number(row_text, min_value=100)
                    if value is not None and value >= 100:
                        unit = cls._detect_emission_unit(row_text)
                        return {
                            'normalized_metric': metric_name,
                            'value': value,
                            'unit': unit,
                            'entity_text': row_text[:100],
                            'context': row_text[:200],
                            'section_type': 'Environmental',
                            'confidence': CONFIDENCE_TABLE_TOTAL if 'total' in row_text.lower() else CONFIDENCE_TABLE_ONLY,
                            'validation_status': 'VALID',
                            'validation_issues': [],
                            'source_type': 'table_reconstructed',
                            'page': row_info['page'],
                        }
        return None
    
    @classmethod
    def _extract_energy(cls, rows):
        for pattern in cls.ENERGY_PATTERNS:
            for row_info in rows:
                row_text = row_info['text']
                if cls._is_rejected(row_text):
                    continue
                if pattern.search(row_text):
                    value = cls._extract_first_valid_number(row_text, min_value=500)
                    if value is not None:
                        unit = cls._detect_energy_unit(row_text)
                        return {
                            'normalized_metric': 'ENERGY_CONSUMPTION',
                            'value': value,
                            'unit': unit,
                            'entity_text': row_text[:100],
                            'context': row_text[:200],
                            'section_type': 'Environmental',
                            'confidence': CONFIDENCE_TABLE_TOTAL if 'total' in row_text.lower() else CONFIDENCE_TABLE_ONLY,
                            'validation_status': 'VALID',
                            'validation_issues': [],
                            'source_type': 'table_reconstructed',
                            'page': row_info['page'],
                        }
        return None
    
    @classmethod
    def _extract_waste(cls, rows):
        for row_info in rows:
            row_text = row_info['text']
            if cls._is_rejected(row_text):
                continue
            if re.search(r'total\s+waste', row_text, re.I):
                value = cls._extract_first_valid_number(row_text, min_value=10)
                if value is not None:
                    return {
                        'normalized_metric': 'WASTE_GENERATED',
                        'value': value,
                        'unit': cls._detect_waste_unit(row_text),
                        'entity_text': row_text[:100],
                        'context': row_text[:200],
                        'section_type': 'Environmental',
                        'confidence': CONFIDENCE_TABLE_TOTAL,
                        'validation_status': 'VALID',
                        'validation_issues': [],
                        'source_type': 'table_reconstructed',
                        'page': row_info['page'],
                    }
        
        total = 0.0
        pages = []
        contexts = []
        seen_categories = set()
        for row_info in rows:
            row_text = row_info['text']
            if cls._is_rejected(row_text):
                continue
            row_lower = row_text.lower()
            matched = False
            for pat in cls.WASTE_PATTERNS:
                if pat.search(row_text):
                    matched = True
                    break
            if not matched:
                continue
            category_key = re.sub(r'[\d,.\s]+', '', row_lower)[:30]
            if category_key in seen_categories:
                continue
            seen_categories.add(category_key)
            value = cls._extract_first_valid_number(row_text, min_value=0.1)
            if value is not None and value > 0:
                total += value
                pages.append(row_info['page'])
                contexts.append(row_text[:80])
        
        if total > 0:
            return {
                'normalized_metric': 'WASTE_GENERATED',
                'value': round(total, 2),
                'unit': 'MT',
                'entity_text': f"Aggregated waste ({len(contexts)} categories)",
                'context': "; ".join(contexts[:3])[:200],
                'section_type': 'Environmental',
                'confidence': CONFIDENCE_TABLE_ONLY,
                'validation_status': 'VALID',
                'validation_issues': ['aggregated_from_categories'],
                'source_type': 'table_reconstructed',
                'page': pages[0] if pages else 0,
            }
        return None
    
    @classmethod
    def _extract_water(cls, rows):
        for row_info in rows:
            row_text = row_info['text']
            if cls._is_rejected(row_text):
                continue
            if re.search(r'total\s+water', row_text, re.I):
                value = cls._extract_first_valid_number(row_text, min_value=100)
                if value is not None:
                    return {
                        'normalized_metric': 'WATER_USAGE',
                        'value': value,
                        'unit': cls._detect_water_unit(row_text),
                        'entity_text': row_text[:100],
                        'context': row_text[:200],
                        'section_type': 'Environmental',
                        'confidence': CONFIDENCE_TABLE_TOTAL,
                        'validation_status': 'VALID',
                        'validation_issues': [],
                        'source_type': 'table_reconstructed',
                        'page': row_info['page'],
                    }
        
        total = 0.0
        pages = []
        contexts = []
        seen_sources = set()
        for row_info in rows:
            row_text = row_info['text']
            if cls._is_rejected(row_text):
                continue
            row_lower = row_text.lower()
            is_water_source = any(
                word in row_lower
                for word in ['groundwater', 'ground water', 'surface water',
                             'third party', 'municipal water', 'tanker water', 'rainwater']
            )
            if not is_water_source:
                continue
            source_key = re.sub(r'[\d,.\s]+', '', row_lower)[:30]
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            value = cls._extract_first_valid_number(row_text, min_value=10)
            if value is not None and value > 0:
                total += value
                pages.append(row_info['page'])
                contexts.append(row_text[:80])
        
        if total > 0:
            return {
                'normalized_metric': 'WATER_USAGE',
                'value': round(total, 2),
                'unit': 'KL',
                'entity_text': f"Aggregated water ({len(contexts)} sources)",
                'context': "; ".join(contexts[:3])[:200],
                'section_type': 'Environmental',
                'confidence': CONFIDENCE_TABLE_ONLY,
                'validation_status': 'VALID',
                'validation_issues': ['aggregated_from_sources'],
                'source_type': 'table_reconstructed',
                'page': pages[0] if pages else 0,
            }
        return None
    
    @staticmethod
    def _detect_emission_unit(text):
        t = text.lower()
        if 'mtco2' in t or 'mt co2' in t: return 'MtCO2e'
        if 'ktco2' in t: return 'ktCO2e'
        if 'tco2' in t or 'tonnes co2' in t or 'metric tonnes' in t: return 'tCO2e'
        if 'co2' in t or 'carbon' in t or 'ghg' in t: return 'tCO2e'
        if 'tonnes' in t or 'metric ton' in t: return 'tCO2e'
        return 'tCO2e'
    
    @staticmethod
    def _detect_energy_unit(text):
        t = text.lower()
        if 'gwh' in t: return 'GWh'
        if 'mwh' in t: return 'MWh'
        if 'kwh' in t: return 'kWh'
        if 'tj' in t: return 'TJ'
        if 'gj' in t: return 'GJ'
        if 'mj' in t: return 'MJ'
        return 'MJ'
    
    @staticmethod
    def _detect_waste_unit(text):
        t = text.lower()
        if 'kg' in t: return 'kg'
        if 'mt' in t or 'metric ton' in t: return 'MT'
        if 'tonnes' in t or 'tons' in t: return 'MT'
        return 'MT'
    
    @staticmethod
    def _detect_water_unit(text):
        t = text.lower()
        if 'kl' in t or 'kilolitre' in t or 'kiloliter' in t: return 'KL'
        if 'm3' in t or 'm³' in t or 'cubic' in t: return 'KL'
        return 'KL'


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "TCS_CORPCS_12062024193403_SEIntBRSRsigned.pdf"
    
    print(f"\n{'='*70}")
    print(f"TABLE RECONSTRUCTOR TEST — {pdf_path}")
    print(f"{'='*70}")
    
    print("\n[1] Extracting text and tables from PDF...")
    pdf_data = extract_text_and_tables(pdf_path)
    print(f"    Pages: {len(pdf_data['pages'])}, Tables: {len(pdf_data['tables'])}")
    
    print("\n[2] Running TableReconstructor...")
    metrics = TableReconstructor.extract_from_pdf(pdf_data)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(metrics)} metrics extracted")
    print(f"{'='*70}")
    
    output = {}
    for m in metrics:
        print(f"\n  {m['normalized_metric']}:")
        print(f"    Value: {m['value']} {m['unit']}")
        print(f"    Source: {m['source_type']} (page {m['page']})")
        print(f"    Confidence: {m['confidence']}")
        print(f"    Context: {m['context'][:120]}...")
        output[m['normalized_metric']] = {
            'value': m['value'],
            'unit': m['unit'],
            'confidence': m['confidence'],
            'source': m['source_type'],
            'page': m['page'],
        }
    
    # Target completeness check
    target = {'SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ENERGY_CONSUMPTION', 
              'WATER_USAGE', 'WASTE_GENERATED', 'ESG_SCORE'}
    found = set(output.keys())
    missing = target - found
    print(f"\n  Completeness: {len(found)}/7 target metrics")
    if missing:
        print(f"  ⚠ Missing: {missing}")
    else:
        print(f"  ✅ All target metrics found!")
    
    # Save output
    out_path = pdf_path.replace('.pdf', '_table_reconstructor.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")
