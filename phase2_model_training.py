"""
PHASE 2: MODEL TRAINING (UPDATED ARCHITECTURE — 11 RAW METRICS)
================================================================

Trains models for the complete 11-layer architecture with 11 target metrics:

1-3. PDF Extraction & Preprocessing (Rule-based)
4. ESG Candidate Filter (Classifier)
5. NER Model (BERT Token Classification) — 11 metrics × 2 (B-/I-) + O = 23 labels
6. Metric Classifier (BERT Sequence Classification) — 11 classes
7-8. Context + Value Extraction (Hybrid: Regex + ML)
9. Unit Normalization (Rule-based)
10. Confidence Scoring (Ensemble)
11. Validation (Rule-based + ML)

Target Metrics (11 RAW METRICS ONLY):
  Environmental (6): SCOPE_1, SCOPE_2, SCOPE_3, ENERGY_CONSUMPTION, 
                     WATER_USAGE, WASTE_GENERATED
  Social (3):        GENDER_DIVERSITY, SAFETY_INCIDENTS, EMPLOYEE_WELLBEING
  Governance (2):    DATA_BREACHES, COMPLAINTS

Dependencies:
pip install transformers torch scikit-learn seqeval
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorForTokenClassification, EarlyStoppingCallback
)
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from dataclasses import dataclass
from pathlib import Path
import re


# TARGET METRICS — all 11 RAW metrics used across the pipeline
TARGET_METRICS = {
    # Environmental (6)
    'SCOPE_1', 'SCOPE_2', 'SCOPE_3',
    'ENERGY_CONSUMPTION', 'WATER_USAGE', 'WASTE_GENERATED',
    # Social (3)
    'GENDER_DIVERSITY', 'SAFETY_INCIDENTS', 'EMPLOYEE_WELLBEING',
    # Governance (2)
    'DATA_BREACHES', 'COMPLAINTS',
}


# ============================================================================
# LAYER 4: ESG CANDIDATE FILTER (NEW)
# ============================================================================

class ESGFilterDataset(Dataset):
    """
    Dataset for training ESG candidate filter
    
    Binary classification: Is this text ESG-related or not?
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Filter dataset has 'is_esg_candidate' label
        print(f"✓ Loaded {len(self.data)} ESG filter samples")
        
        # Check distribution
        esg_count = sum(1 for s in self.data if s.get('is_esg_candidate', True))
        print(f"  ESG: {esg_count}, Non-ESG: {len(self.data) - esg_count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        
        # Binary label: ESG (1) or Non-ESG (0)
        label = 1 if sample.get('is_esg_candidate', True) else 0
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ESGFilterTrainer:
    """Train ESG candidate filter (Layer 4)"""
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',  # Faster for filtering
        output_dir: str = './models/esg_filter'
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def train(
        self,
        train_path: str,
        val_path: str,
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 3e-5
    ):
        """Train ESG filter"""
        
        print("\n" + "="*70)
        print("TRAINING ESG CANDIDATE FILTER (LAYER 4)")
        print("="*70)
        
        train_dataset = ESGFilterDataset(train_path, self.tokenizer)
        val_dataset = ESGFilterDataset(val_path, self.tokenizer)
        
        # Binary classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            save_total_limit=2
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary'
            )
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("\nStarting training...")
        trainer.train()
        
        trainer.save_model(str(self.output_dir / 'final'))
        self.tokenizer.save_pretrained(str(self.output_dir / 'final'))
        
        print(f"\n✓ ESG Filter model saved to {self.output_dir / 'final'}")


# ============================================================================
# LAYER 5: NER MODEL
# ============================================================================

class ESGNERDataset(Dataset):
    """Dataset for NER model with architecture support"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter to only ESG candidates for training efficiency
        self.data = [s for s in self.data if s.get('is_esg_candidate', True)]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.label_list = self._build_label_list()
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        print(f"✓ Loaded {len(self.data)} NER samples (ESG only)")
        print(f"✓ Found {len(self.label_list)} unique labels")
    
    def _build_label_list(self) -> List[str]:
        """Build label list from target metrics only"""
        labels = set(['O'])
        for metric in TARGET_METRICS:
            labels.add(f'B-{metric}')
            labels.add(f'I-{metric}')
        return sorted(list(labels))
    
    def _filter_tags_to_target(self, tags: List[str]) -> List[str]:
        """Relabel non-target metric tags to 'O'"""
        filtered = []
        for tag in tags:
            if tag == 'O':
                filtered.append('O')
            else:
                # Extract metric name from B-METRIC or I-METRIC
                prefix, metric = tag.split('-', 1)
                if metric in TARGET_METRICS:
                    filtered.append(tag)
                else:
                    filtered.append('O')
        return filtered
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        tags = self._filter_tags_to_target(sample['tags'])
        
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = self._align_labels(tokenized.word_ids(), tags)
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _align_labels(self, word_ids: List[Optional[int]], tags: List[str]) -> List[int]:
        """Align labels with subwords - CRITICAL for BERT"""
        labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # Special token
            elif word_id != previous_word_id:
                labels.append(self.label2id[tags[word_id]])  # First subword
            else:
                labels.append(-100)  # Subsequent subword - ignore
            
            previous_word_id = word_id
        
        return labels


class NERTrainer:
    """Train BERT-based NER model (Layer 5)"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        output_dir: str = './models/ner_model'
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def train(
        self,
        train_path: str,
        val_path: str,
        num_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 5e-5
    ):
        """Train NER model"""
        
        print("\n" + "="*70)
        print("TRAINING NER MODEL (LAYER 5)")
        print("="*70)
        
        train_dataset = ESGNERDataset(train_path, self.tokenizer)
        val_dataset = ESGNERDataset(val_path, self.tokenizer)
        
        num_labels = len(train_dataset.label_list)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=train_dataset.id2label,
            label2id=train_dataset.label2id
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=100,
            save_total_limit=3
        )
        
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=2)
            
            true_labels = [[train_dataset.id2label[l] for l in label if l != -100] 
                          for label in labels]
            true_predictions = [[train_dataset.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                               for prediction, label in zip(predictions, labels)]
            
            return {
                'f1': seq_f1_score(true_labels, true_predictions),
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("\nStarting training...")
        trainer.train()
        
        trainer.save_model(str(self.output_dir / 'final'))
        self.tokenizer.save_pretrained(str(self.output_dir / 'final'))
        
        with open(self.output_dir / 'label_mappings.json', 'w') as f:
            json.dump({
                'label2id': train_dataset.label2id,
                'id2label': train_dataset.id2label
            }, f, indent=2)
        
        print(f"\n✓ NER model saved to {self.output_dir / 'final'}")


# ============================================================================
# LAYER 6: METRIC CLASSIFIER
# ============================================================================

class MetricClassificationDataset(Dataset):
    """Dataset for metric classification with section awareness"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter to target metrics only
        self.data = [d for d in self.data if d['label'] in TARGET_METRICS]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.label_list = sorted(list(set(d['label'] for d in self.data)))
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        print(f"✓ Loaded {len(self.data)} classification samples")
        print(f"✓ Found {len(self.label_list)} unique classes")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # NEW: Include section type as context
        section_type = sample.get('section_type', '')
        text = f"[{section_type}] {sample['metric_text']} [SEP] {sample['context']}"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id[sample['label']], dtype=torch.long)
        }


class ClassifierTrainer:
    """Train metric classification model (Layer 6)"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        output_dir: str = './models/classifier'
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def train(
        self,
        train_path: str,
        val_path: str,
        num_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 3e-5
    ):
        """Train classification model"""
        
        print("\n" + "="*70)
        print("TRAINING METRIC CLASSIFIER (LAYER 6)")
        print("="*70)
        
        train_dataset = MetricClassificationDataset(train_path, self.tokenizer)
        val_dataset = MetricClassificationDataset(val_path, self.tokenizer)
        
        num_labels = len(train_dataset.label_list)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            logging_steps=50,
            save_total_limit=3
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1_weighted': f1_score(labels, predictions, average='weighted'),
                'f1_macro': f1_score(labels, predictions, average='macro')
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("\nStarting training...")
        trainer.train()
        
        trainer.save_model(str(self.output_dir / 'final'))
        self.tokenizer.save_pretrained(str(self.output_dir / 'final'))
        
        with open(self.output_dir / 'label_mappings.json', 'w') as f:
            json.dump({
                'label2id': train_dataset.label2id,
                'id2label': train_dataset.id2label
            }, f, indent=2)
        
        print(f"\n✓ Classifier model saved to {self.output_dir / 'final'}")


# ============================================================================
# LAYER 7-8: CONTEXT-AWARE VALUE EXTRACTION
# ============================================================================

class ValueExtractor:
    """
    Hybrid value extraction (Layers 7-8)
    
    Uses context windows from Phase 1 data
    """
    
    def __init__(self):
        self.patterns = [
            (r'([\d,]+\.\d+)\s*([a-zA-Z/%³]+)', 'full'),
            (r'([\d,]+)\s*([a-zA-Z/%³]+)', 'int'),
            (r'(\d+\.\d+)', 'float_only'),
            (r'(\d+)', 'int_only')
        ]
        
        self.unit_map = {
            # Emissions
            'tonnes': 'tCO2e', 'tons': 'tCO2e', 'tco2e': 'tCO2e',
            'mtco2e': 'Mt CO2e', 'million tonnes': 'Mt CO2e',
            # Energy
            'gwh': 'GWh', 'mwh': 'MWh', 'kwh': 'kWh',
            # Water
            'm3': 'm³', 'cubic meters': 'm³', 'ml': 'ML',
            # Percentage
            '%': '%', 'percent': '%', 'percentage': '%',
            # Others
            'fte': 'FTE', 'employees': 'employees',
            'hours': 'hours', 'hours/employee': 'hours/employee'
        }
    
    def extract_from_context(self, context: str) -> List[Dict]:
        """Extract (value, unit) pairs from context window"""
        results = []
        
        for pattern, pattern_type in self.patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    unit = ""
                    if len(match.groups()) >= 2:
                        unit_raw = match.group(2)
                        unit = self.unit_map.get(unit_raw.lower(), unit_raw)
                    
                    # Confidence based on pattern completeness
                    confidence_map = {
                        'full': 0.95,
                        'int': 0.90,
                        'float_only': 0.70,
                        'int_only': 0.60
                    }
                    
                    results.append({
                        'value': value,
                        'unit': unit,
                        'confidence': confidence_map[pattern_type],
                        'method': 'regex'
                    })
                
                except (ValueError, IndexError):
                    continue
        
        # Deduplicate
        seen = set()
        deduped = []
        for r in results:
            key = (r['value'], r['unit'])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        
        return sorted(deduped, key=lambda x: x['confidence'], reverse=True)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Train all models with full architecture support"""
    
    print("\n" + "="*70)
    print("ESG EXTRACTION MODELS - COMPLETE TRAINING PIPELINE")
    print("="*70)
    print("\nArchitecture Layers Trained:")
    print("  • Layer 4: ESG Candidate Filter")
    print("  • Layer 5: NER Model")
    print("  • Layer 6: Metric Classifier")
    print("  • Layer 7-8: Context-Aware Value Extraction")
    print("="*70)
    
    data_dir = Path('./processed_data')
    if not data_dir.exists():
        print("\n❌ ERROR: Processed data not found!")
        print("   Please run phase1_data_transformation.py first.")
        return
    
    # 1. Train ESG Candidate Filter (Layer 4) - NEW
    print("\n" + "="*70)
    print("STEP 1/3: Training ESG Candidate Filter")
    print("="*70)
    
    filter_trainer = ESGFilterTrainer(
        model_name='distilbert-base-uncased',  # Faster
        output_dir='./models/esg_filter'
    )
    
    filter_trainer.train(
        train_path=str(data_dir / 'ner_train.json'),  # Has is_esg_candidate labels
        val_path=str(data_dir / 'ner_val.json'),
        num_epochs=3,
        batch_size=32
    )
    
    # 2. Train NER Model (Layer 5)
    print("\n" + "="*70)
    print("STEP 2/3: Training NER Model")
    print("="*70)
    
    ner_trainer = NERTrainer(
        model_name='bert-base-uncased',
        output_dir='./models/ner_model'
    )
    
    ner_trainer.train(
        train_path=str(data_dir / 'ner_train.json'),
        val_path=str(data_dir / 'ner_val.json'),
        num_epochs=10,
        batch_size=16
    )
    
    # 3. Train Metric Classifier (Layer 6)
    print("\n" + "="*70)
    print("STEP 3/3: Training Metric Classifier")
    print("="*70)
    
    classifier_trainer = ClassifierTrainer(
        model_name='bert-base-uncased',
        output_dir='./models/classifier'
    )
    
    classifier_trainer.train(
        train_path=str(data_dir / 'classification_train.json'),
        val_path=str(data_dir / 'classification_val.json'),
        num_epochs=5,
        batch_size=32
    )
    
    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY")
    print("="*70)
    print("\nModels saved to:")
    print("  • ./models/esg_filter/final/ (Layer 4)")
    print("  • ./models/ner_model/final/ (Layer 5)")
    print("  • ./models/classifier/final/ (Layer 6)")
    print("\nOther layers:")
    print("  • Layers 1-3: Rule-based (PDF extraction, preprocessing)")
    print("  • Layers 7-8: Hybrid regex + ML (see ValueExtractor class)")
    print("  • Layer 9: Rule-based (unit normalization)")
    print("  • Layer 10: Ensemble (confidence scoring)")
    print("  • Layer 11: Rule-based + validation data")
    print("\nNext steps:")
    print("  1. Evaluate models on test set")
    print("  2. Test on real PDF documents")
    print("  3. Integrate into production pipeline")


if __name__ == "__main__":
    main()
