"""
PHASE 1: DATA TRANSFORMATION (UPDATED ARCHITECTURE)
====================================================

Transforms synthetic ESG dataset into production-ready ML formats with
FULL SUPPORT for the 11-layer architecture:

1. Text & Table Extraction
2. Document Structuring Layer ← NEW
3. Text Preprocessing
4. Candidate ESG Filter ← NEW
5. NER Model
6. Metric Classifier
7. Context Window Extraction ← NEW
8. Value Extraction
9. Unit Normalization
10. Confidence Scoring
11. Validation Layer ← NEW

Key Features:
- Negative samples (non-ESG metrics)
- ESG filtering examples
- Validation ground truth
- Context windows for value extraction
- Document section simulation
"""

import json
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import random
from dataclasses import dataclass
import numpy as np
from pathlib import Path


# TARGET METRICS — only these 7 metrics are used
TARGET_METRICS: Set[str] = {
    'SCOPE_1', 'SCOPE_2', 'SCOPE_3',
    'ENERGY_CONSUMPTION', 'WATER_USAGE', 'WASTE_GENERATED',
    'ESG_SCORE',
}


@dataclass
class EntitySpan:
    """Represents a metric entity in text"""
    start: int
    end: int
    text: str
    label: str
    metric_id: int
    confidence: float = 1.0
    section_type: str = "unknown"  # NEW: Environmental, Social, Governance


@dataclass
class ValidationExample:
    """Ground truth for validation layer"""
    metric: str
    normalized_metric: str
    value: float
    unit: str
    is_valid: bool
    validation_issues: List[str]


class ESGDataTransformer:
    """
    Production-grade data transformation with full architecture support
    """
    
    def __init__(self, dataset_path: str, verbose: bool = True):
        self.verbose = verbose
        self._log(f"Loading dataset from {dataset_path}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self._log(f"✓ Loaded {len(self.raw_data):,} samples")
        
        # Initialize ESG filter keywords (Layer 4)
        self._init_esg_keywords()
        
        # Initialize validation rules (Layer 11)
        self._init_validation_rules()
        
        self._analyze_dataset()
    
    def _init_esg_keywords(self):
        """Initialize ESG candidate filter keywords"""
        self.esg_keywords = {
            'environmental': [
                'emissions', 'carbon', 'co2', 'ghg', 'scope', 'energy', 'renewable',
                'water', 'waste', 'recycling', 'climate', 'environmental', 'sustainability'
            ],
            'social': [
                'employees', 'workforce', 'diversity', 'gender', 'training', 'safety',
                'injury', 'accident', 'turnover', 'attrition', 'health', 'labor'
            ],
            'governance': [
                'board', 'directors', 'governance', 'ethics', 'compliance', 'esg',
                'transparency', 'risk', 'audit', 'independence'
            ]
        }
        
        self.non_esg_keywords = {
            'financial': [
                'revenue', 'profit', 'earnings', 'ebitda', 'cash flow', 'dividend',
                'stock price', 'market cap', 'valuation', 'debt', 'equity'
            ],
            'operational': [
                'production', 'sales', 'customers', 'market share', 'units sold'
            ]
        }
    
    def _init_validation_rules(self):
        """Initialize validation rules for Layer 11"""
        self.validation_ranges = {
            'SCOPE_1': (0, 100_000_000),
            'SCOPE_2': (0, 100_000_000),
            'SCOPE_3': (0, 500_000_000),
            'ENERGY_CONSUMPTION': (0, 50_000_000),
            'WATER_USAGE': (0, 100_000_000),
            'WASTE_GENERATED': (0, 10_000_000),
            'ESG_SCORE': (0, 100),
        }
    
    def _log(self, message: str):
        """Logging utility"""
        if self.verbose:
            print(message)
    
    def _analyze_dataset(self):
        """Comprehensive dataset analysis"""
        metric_counts = Counter()
        category_counts = Counter()
        
        for sample in self.raw_data:
            for metric in sample['metrics']:
                metric_counts[metric['normalized_metric']] += 1
                category_counts[metric['category']] += 1
        
        self._log("\n" + "="*70)
        self._log("DATASET ANALYSIS")
        self._log("="*70)
        self._log(f"Total samples: {len(self.raw_data):,}")
        self._log(f"Unique metrics: {len(metric_counts)}")
        
        self._log(f"\nCategory distribution:")
        for cat, count in category_counts.items():
            self._log(f"  {cat}: {count:,} ({count/sum(category_counts.values())*100:.1f}%)")
    
    def classify_section_type(self, text: str) -> str:
        """
        Layer 2: Document Structuring - Classify section type
        """
        text_lower = text.lower()
        
        scores = {
            'Environmental': sum(text_lower.count(kw) for kw in self.esg_keywords['environmental']),
            'Social': sum(text_lower.count(kw) for kw in self.esg_keywords['social']),
            'Governance': sum(text_lower.count(kw) for kw in self.esg_keywords['governance'])
        }
        
        if max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def is_esg_candidate(self, text: str) -> bool:
        """
        Layer 4: ESG Candidate Filter
        Returns True if text likely contains ESG metrics
        """
        text_lower = text.lower()
        
        # Count ESG keywords
        esg_count = sum(
            sum(text_lower.count(kw) for kw in keywords)
            for keywords in self.esg_keywords.values()
        )
        
        # Count non-ESG keywords
        non_esg_count = sum(
            sum(text_lower.count(kw) for kw in keywords)
            for keywords in self.non_esg_keywords.values()
        )
        
        total = esg_count + non_esg_count
        if total == 0:
            return False
        
        # ESG keywords should dominate
        return (esg_count / total) >= 0.5
    
    def validate_metric(self, normalized_metric: str, value: float) -> Tuple[bool, List[str]]:
        """
        Layer 11: Validation Layer
        Returns (is_valid, issues_list)
        """
        issues = []
        
        if normalized_metric not in self.validation_ranges:
            issues.append(f"Unknown metric: {normalized_metric}")
            return False, issues
        
        min_val, max_val = self.validation_ranges[normalized_metric]
        
        if value < min_val:
            issues.append(f"Value {value} below minimum {min_val}")
            return False, issues
        
        if value > max_val:
            issues.append(f"Value {value} above maximum {max_val}")
            return False, issues
        
        return True, []
    
    def find_metric_spans(
        self, 
        text: str, 
        metric_name: str, 
        metric_id: int, 
        normalized_metric: str,
        section_type: str = "unknown"
    ) -> List[EntitySpan]:
        """
        Layer 5: NER - Find all occurrences with section context
        """
        spans = []
        text_lower = text.lower()
        metric_lower = metric_name.lower()
        
        start = 0
        while True:
            idx = text_lower.find(metric_lower, start)
            if idx == -1:
                break
            
            end = idx + len(metric_name)
            
            # Validate word boundaries
            valid_start = (idx == 0 or not text[idx-1].isalnum())
            valid_end = (end >= len(text) or not text[end].isalnum())
            
            if valid_start and valid_end:
                spans.append(EntitySpan(
                    start=idx,
                    end=end,
                    text=text[idx:end],
                    label=normalized_metric,
                    metric_id=metric_id,
                    confidence=1.0,
                    section_type=section_type
                ))
            
            start = idx + 1
        
        return spans
    
    def extract_context_window(
        self, 
        text: str, 
        entity_start: int, 
        entity_end: int,
        window_size: int = 150
    ) -> str:
        """
        Layer 7: Context Window Extraction
        """
        context_start = max(0, entity_start - window_size)
        context_end = min(len(text), entity_end + window_size)
        return text[context_start:context_end]
    
    def _resolve_overlaps(self, spans: List[EntitySpan]) -> List[EntitySpan]:
        """Resolve overlapping entity spans"""
        if not spans:
            return []
        
        sorted_spans = sorted(spans, key=lambda x: x.start)
        non_overlapping = []
        current_span = sorted_spans[0]
        
        for next_span in sorted_spans[1:]:
            if next_span.start < current_span.end:
                current_length = current_span.end - current_span.start
                next_length = next_span.end - next_span.start
                if next_length > current_length:
                    current_span = next_span
            else:
                non_overlapping.append(current_span)
                current_span = next_span
        
        non_overlapping.append(current_span)
        return non_overlapping
    
    def _create_bio_tags(
        self, 
        text: str, 
        spans: List[EntitySpan]
    ) -> Tuple[List[str], List[str]]:
        """Create BIO tags for tokens"""
        tokens = []
        token_positions = []
        
        for match in re.finditer(r'\S+', text):
            tokens.append(match.group())
            token_positions.append((match.start(), match.end()))
        
        tags = ['O'] * len(tokens)
        
        for span in spans:
            is_first_token = True
            
            for token_idx, (tok_start, tok_end) in enumerate(token_positions):
                if tok_start >= span.end:
                    break
                if tok_end <= span.start:
                    continue
                
                if is_first_token:
                    tags[token_idx] = f'B-{span.label}'
                    is_first_token = False
                else:
                    tags[token_idx] = f'I-{span.label}'
        
        return tokens, tags
    
    def generate_negative_samples(self, num_samples: int = 5000) -> List[Dict]:
        """
        Layer 4 Support: Generate negative (non-ESG) samples
        
        CRITICAL for training the ESG candidate filter and reducing false positives
        """
        self._log(f"\nGenerating {num_samples:,} negative samples...")
        
        negative_samples = []
        
        # Financial metric templates
        financial_templates = [
            "Revenue for fiscal year {year} was ${value} million.",
            "Net income increased to ${value} million, up {pct}% year-over-year.",
            "EBITDA reached ${value} million in Q{quarter}.",
            "Operating margin improved to {value}% from {prev}%.",
            "Earnings per share (EPS) were ${value} for the year.",
            "Total assets grew to ${value} billion.",
            "Cash and equivalents: ${value} million at year end.",
            "Debt-to-equity ratio stood at {value}.",
            "Return on equity (ROE) was {value}%.",
            "Stock price closed at ${value} per share."
        ]
        
        # Operational metric templates
        operational_templates = [
            "Total units sold: {value} million.",
            "Market share increased to {value}%.",
            "Customer satisfaction score: {value} out of 100.",
            "Production volume reached {value} units.",
            "Average order value: ${value}.",
            "Conversion rate improved to {value}%.",
            "Monthly active users: {value} million.",
            "Churn rate decreased to {value}%."
        ]
        
        all_templates = financial_templates + operational_templates
        
        for i in range(num_samples):
            template = random.choice(all_templates)
            
            # Generate realistic values
            value = round(random.uniform(1, 10000), 2)
            pct = round(random.uniform(1, 50), 1)
            prev = round(random.uniform(1, 30), 1)
            year = random.randint(2018, 2023)
            quarter = random.randint(1, 4)
            
            text = template.format(
                value=value, 
                pct=pct, 
                prev=prev, 
                year=year, 
                quarter=quarter
            )
            
            negative_samples.append({
                'id': f'negative_{i}',
                'text': text,
                'tokens': text.split(),
                'tags': ['O'] * len(text.split()),
                'entities': [],
                'year': year,
                'is_negative': True,
                'is_esg_candidate': False,  # NEW: Ground truth for filter
                'section_type': 'financial'
            })
        
        self._log(f"✓ Generated {len(negative_samples):,} negative samples")
        return negative_samples
    
    def generate_validation_examples(self) -> List[Dict]:
        """
        Layer 11 Support: Generate validation ground truth
        
        Includes both valid and invalid examples for testing validation rules
        """
        self._log("\nGenerating validation examples...")
        
        validation_examples = []
        
        # Valid examples (within range)
        for metric, (min_val, max_val) in self.validation_ranges.items():
            for _ in range(3):
                value = random.uniform(min_val, max_val)
                is_valid, issues = self.validate_metric(metric, value)
                
                validation_examples.append({
                    'metric': metric,
                    'value': value,
                    'is_valid': is_valid,
                    'issues': issues,
                    'test_type': 'valid_range'
                })
        
        # Invalid examples (out of range)
        for metric, (min_val, max_val) in self.validation_ranges.items():
            # Below minimum
            value = min_val - random.uniform(100, 1000)
            is_valid, issues = self.validate_metric(metric, value)
            validation_examples.append({
                'metric': metric,
                'value': value,
                'is_valid': is_valid,
                'issues': issues,
                'test_type': 'below_range'
            })
            
            # Above maximum
            value = max_val + random.uniform(100, 10000)
            is_valid, issues = self.validate_metric(metric, value)
            validation_examples.append({
                'metric': metric,
                'value': value,
                'is_valid': is_valid,
                'issues': issues,
                'test_type': 'above_range'
            })
        
        self._log(f"✓ Generated {len(validation_examples):,} validation examples")
        return validation_examples
    
    def convert_to_ner_format(self, include_negatives: bool = True) -> List[Dict]:
        """
        Convert to NER format with full architecture support
        
        NEW: Includes negative samples and section types
        """
        self._log("\n" + "="*70)
        self._log("CONVERTING TO NER FORMAT (WITH ARCHITECTURE SUPPORT)")
        self._log("="*70)
        
        ner_dataset = []
        
        # Process positive (ESG) samples
        for sample_idx, sample in enumerate(self.raw_data):
            text = sample['text']
            
            # Layer 2: Classify section type
            section_type = self.classify_section_type(text)
            
            # Layer 4: Check if ESG candidate
            is_esg = self.is_esg_candidate(text)
            
            # Find all entity spans — only for target metrics
            all_spans = []
            for metric_idx, metric in enumerate(sample['metrics']):
                if metric['normalized_metric'] not in TARGET_METRICS:
                    continue
                spans = self.find_metric_spans(
                    text,
                    metric['metric'],
                    metric_idx,
                    metric['normalized_metric'],
                    section_type
                )
                all_spans.extend(spans)
            
            # Resolve overlaps
            all_spans.sort(key=lambda x: x.start)
            non_overlapping_spans = self._resolve_overlaps(all_spans)
            
            # Create BIO tags
            tokens, tags = self._create_bio_tags(text, non_overlapping_spans)
            
            # Build entities with context windows (Layer 7)
            entities = []
            for span in non_overlapping_spans:
                context = self.extract_context_window(text, span.start, span.end)
                entities.append({
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "text": span.text,
                    "context_window": context,  # NEW
                    "section_type": span.section_type  # NEW
                })
            
            ner_dataset.append({
                "id": f"sample_{sample_idx}",
                "text": text,
                "tokens": tokens,
                "tags": tags,
                "entities": entities,
                "year": sample.get('year'),
                "is_esg_candidate": is_esg,  # NEW
                "section_type": section_type  # NEW
            })
        
        # Add negative samples (Layer 4 training data)
        if include_negatives:
            negatives = self.generate_negative_samples(
                num_samples=int(len(ner_dataset) * 0.15)  # 15% negatives
            )
            ner_dataset.extend(negatives)
            random.shuffle(ner_dataset)  # Mix positives and negatives
        
        self._log(f"✓ Created {len(ner_dataset):,} NER samples")
        self._log(f"  Positive (ESG): {len([s for s in ner_dataset if not s.get('is_negative')])}")
        self._log(f"  Negative (non-ESG): {len([s for s in ner_dataset if s.get('is_negative')])}")
        
        return ner_dataset
    
    def convert_to_classification_format(self) -> List[Dict]:
        """Convert to metric classification format"""
        self._log("\n" + "="*70)
        self._log("CONVERTING TO CLASSIFICATION FORMAT")
        self._log("="*70)
        
        classification_dataset = []
        
        for sample_idx, sample in enumerate(self.raw_data):
            text = sample['text']
            section_type = self.classify_section_type(text)
            
            for metric_idx, metric in enumerate(sample['metrics']):
                metric_name = metric['metric']
                normalized_metric = metric['normalized_metric']
                
                # Skip non-target metrics
                if normalized_metric not in TARGET_METRICS:
                    continue
                
                # Find metric in text
                spans = self.find_metric_spans(
                    text, metric_name, metric_idx, normalized_metric, section_type
                )
                
                if not spans:
                    context = text
                    metric_text = metric_name
                else:
                    span = spans[0]
                    # Layer 7: Use context window
                    context = self.extract_context_window(text, span.start, span.end)
                    metric_text = span.text
                
                classification_dataset.append({
                    "id": f"sample_{sample_idx}_metric_{metric_idx}",
                    "metric_text": metric_text,
                    "context": context,
                    "label": normalized_metric,
                    "category": metric['category'],
                    "section_type": section_type  # NEW
                })
        
        self._log(f"✓ Created {len(classification_dataset):,} classification samples")
        return classification_dataset
    
    def convert_to_relation_extraction_format(self) -> List[Dict]:
        """Convert to relation extraction format with validation"""
        self._log("\n" + "="*70)
        self._log("CONVERTING TO RELATION EXTRACTION FORMAT")
        self._log("="*70)
        
        relation_dataset = []
        
        for sample_idx, sample in enumerate(self.raw_data):
            text = sample['text']
            
            for metric_idx, metric in enumerate(sample['metrics']):
                metric_name = metric['metric']
                normalized_metric = metric['normalized_metric']
                value = metric['value']
                unit = metric.get('unit', '')
                
                # Skip non-target metrics
                if normalized_metric not in TARGET_METRICS:
                    continue
                
                # Layer 11: Validate metric
                is_valid, issues = self.validate_metric(normalized_metric, value)
                
                # Find metric spans
                metric_spans = self.find_metric_spans(
                    text, metric_name, metric_idx, normalized_metric
                )
                
                if not metric_spans:
                    continue
                
                for span in metric_spans:
                    # Layer 7: Extract context window
                    context = self.extract_context_window(text, span.start, span.end, window_size=150)
                    
                    # Find value in context
                    value_str = str(value)
                    value_patterns = [
                        re.escape(value_str),
                        re.escape(value_str.replace('.0', '')),
                        re.escape(f"{value:,}"),
                    ]
                    
                    value_match = None
                    for pattern in value_patterns:
                        value_match = re.search(pattern, context)
                        if value_match:
                            break
                    
                    if not value_match:
                        continue
                    
                    value_start = value_match.start()
                    value_end = value_match.end()
                    
                    # Find unit
                    unit_start = -1
                    unit_end = -1
                    
                    if unit:
                        unit_pattern = re.escape(unit)
                        search_start = max(0, value_end)
                        search_end = min(len(context), value_end + 30)
                        unit_search = context[search_start:search_end]
                        unit_match = re.search(unit_pattern, unit_search, re.IGNORECASE)
                        
                        if unit_match:
                            unit_start = search_start + unit_match.start()
                            unit_end = search_start + unit_match.end()
                    
                    relation_dataset.append({
                        "id": f"sample_{sample_idx}_metric_{metric_idx}",
                        "text": context,
                        "metric": {
                            "text": span.text,
                            "label": normalized_metric,
                            "start": span.start - (span.start - 150 if span.start > 150 else 0),
                            "end": span.end - (span.start - 150 if span.start > 150 else 0)
                        },
                        "value": {
                            "text": value_match.group(),
                            "numeric": value,
                            "start": value_start,
                            "end": value_end
                        },
                        "unit": {
                            "text": unit,
                            "start": unit_start,
                            "end": unit_end
                        } if unit and unit_start >= 0 else None,
                        "is_valid": is_valid,  # NEW: Validation ground truth
                        "validation_issues": issues  # NEW
                    })
        
        self._log(f"✓ Created {len(relation_dataset):,} relation extraction samples")
        valid_count = sum(1 for s in relation_dataset if s['is_valid'])
        self._log(f"  Valid: {valid_count}, Invalid: {len(relation_dataset) - valid_count}")
        
        return relation_dataset
    
    def create_train_val_test_splits(
        self,
        dataset: List[Dict],
        stratify_key: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """Create stratified splits"""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        if stratify_key:
            label_to_indices = defaultdict(list)
            
            for idx, sample in enumerate(dataset):
                if stratify_key in sample:
                    label = sample[stratify_key]
                elif 'entities' in sample and sample['entities']:
                    label = sample['entities'][0]['label']
                else:
                    label = 'NO_LABEL'
                
                label_to_indices[label].append(idx)
            
            train_indices = []
            val_indices = []
            test_indices = []
            
            for label, indices in label_to_indices.items():
                random.shuffle(indices)
                n = len(indices)
                
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)
                
                train_indices.extend(indices[:train_end])
                val_indices.extend(indices[train_end:val_end])
                test_indices.extend(indices[val_end:])
            
            random.shuffle(train_indices)
            random.shuffle(val_indices)
            random.shuffle(test_indices)
        
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            n = len(indices)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
        
        splits = {
            'train': [dataset[i] for i in train_indices],
            'val': [dataset[i] for i in val_indices],
            'test': [dataset[i] for i in test_indices]
        }
        
        self._log(f"\n{'='*70}")
        self._log("DATASET SPLITS")
        self._log(f"{'='*70}")
        self._log(f"Train: {len(splits['train']):,} ({len(splits['train'])/len(dataset)*100:.1f}%)")
        self._log(f"Val:   {len(splits['val']):,} ({len(splits['val'])/len(dataset)*100:.1f}%)")
        self._log(f"Test:  {len(splits['test']):,} ({len(splits['test'])/len(dataset)*100:.1f}%)")
        
        return splits
    
    def save_datasets(self, output_dir: str = './processed_data'):
        """Save all processed datasets with architecture support"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._log(f"\n{'='*70}")
        self._log("SAVING PROCESSED DATASETS (WITH FULL ARCHITECTURE)")
        self._log(f"{'='*70}")
        
        # Convert to all formats
        ner_data = self.convert_to_ner_format(include_negatives=True)
        classification_data = self.convert_to_classification_format()
        relation_data = self.convert_to_relation_extraction_format()
        
        # Generate validation examples
        validation_data = self.generate_validation_examples()
        
        # Create splits
        ner_splits = self.create_train_val_test_splits(ner_data)
        classification_splits = self.create_train_val_test_splits(
            classification_data, 
            stratify_key='label'
        )
        relation_splits = self.create_train_val_test_splits(relation_data)
        
        # Save NER data
        for split_name, split_data in ner_splits.items():
            filepath = output_path / f'ner_{split_name}.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            self._log(f"✓ Saved: {filepath} ({len(split_data):,} samples)")
        
        # Save classification data
        for split_name, split_data in classification_splits.items():
            filepath = output_path / f'classification_{split_name}.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            self._log(f"✓ Saved: {filepath} ({len(split_data):,} samples)")
        
        # Save relation data
        for split_name, split_data in relation_splits.items():
            filepath = output_path / f'relation_{split_name}.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            self._log(f"✓ Saved: {filepath} ({len(split_data):,} samples)")
        
        # Save validation data
        filepath = output_path / 'validation_examples.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        self._log(f"✓ Saved: {filepath} ({len(validation_data):,} examples)")
        
        self._log(f"\n✓ All datasets saved to {output_dir}")
        self._log("\nNew architecture features included:")
        self._log("  ✓ ESG candidate filtering (positive + negative samples)")
        self._log("  ✓ Section type classification (Environmental/Social/Governance)")
        self._log("  ✓ Context windows for value extraction")
        self._log("  ✓ Validation ground truth (valid + invalid examples)")


def main():
    """Example usage"""
    transformer = ESGDataTransformer('esg_dataset.json', verbose=True)
    transformer.save_datasets('./processed_data')
    
    print("\n" + "="*70)
    print("✓ DATA TRANSFORMATION COMPLETE (UPDATED ARCHITECTURE)")
    print("="*70)
    print("\nDatasets now include:")
    print("  • Negative samples for ESG filtering")
    print("  • Section type labels")
    print("  • Context windows")
    print("  • Validation examples")


if __name__ == "__main__":
    main()
