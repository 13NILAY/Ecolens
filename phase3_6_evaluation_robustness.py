"""
PHASES 3-6: EVALUATION, ROBUSTNESS, AND PRODUCTION (UPDATED ARCHITECTURE)
==========================================================================

Complete evaluation and deployment for all 11 layers:

Layer 4: ESG Candidate Filter evaluation
Layer 5: NER model evaluation
Layer 6: Metric Classifier evaluation
Layer 7-11: Value extraction, validation, end-to-end testing

Plus:
- Real-world PDF noise simulation
- Confidence calibration
- Production deployment guide
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path
import re
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)


# ============================================================================
# PHASE 5: REAL-WORLD ROBUSTNESS
# ============================================================================

class SyntheticToRealBridge:
    """Bridge synthetic data to real PDF scenarios"""
    
    def __init__(self):
        self.pdf_noise_patterns = {
            'broken_words': [
                (r'(\w{4,})', lambda m: m.group(1)[:len(m.group(1))//2] + '-\n' + m.group(1)[len(m.group(1))//2:])
            ],
            'table_artifacts': ['|', '─', '┌', '└', '├', '┤', '•'],
            'encoding_issues': {
                'â€™': "'", 'â€"': '—', 'Â': '', 'â€¢': '•'
            },
            'ocr_errors': {
                'O': '0', 'l': '1', 'S': '5', 'B': '8'
            }
        }
    
    def add_pdf_noise(self, text: str, noise_level: float = 0.3) -> str:
        """
        Add realistic PDF extraction noise
        
        Args:
            text: Clean text
            noise_level: 0.0-1.0 probability of noise
        """
        import random
        
        if random.random() > noise_level:
            return text
        
        # 1. Broken word hyphenation (20% chance)
        if random.random() < 0.2:
            words = text.split()
            if len(words) > 3:
                idx = random.randint(0, len(words) - 2)
                word = words[idx]
                if len(word) > 6:
                    split_pos = len(word) // 2
                    words[idx] = word[:split_pos] + '-\n' + word[split_pos:]
                text = ' '.join(words)
        
        # 2. Table artifacts (10% chance)
        if random.random() < 0.1:
            artifact = random.choice(self.pdf_noise_patterns['table_artifacts'])
            text = artifact + ' ' + text
        
        # 3. Encoding issues (15% chance)
        if random.random() < 0.15:
            for bad, good in self.pdf_noise_patterns['encoding_issues'].items():
                if random.random() < 0.5 and good in text:
                    text = text.replace(good, bad)
        
        # 4. OCR errors in numbers (5% chance)
        if random.random() < 0.05:
            for char, replacement in self.pdf_noise_patterns['ocr_errors'].items():
                if char in text and random.random() < 0.3:
                    # Replace one occurrence
                    text = text.replace(char, replacement, 1)
        
        # 5. Extra whitespace (10% chance)
        if random.random() < 0.1:
            text = re.sub(r'(\w)\s+(\w)', r'\1  \2', text)
        
        return text
    
    def augment_dataset(
        self,
        dataset_path: str,
        output_path: str,
        noise_level: float = 0.3
    ):
        """Augment dataset with PDF noise"""
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"\nAugmenting dataset: {dataset_path}")
        print(f"Original samples: {len(data):,}")
        
        augmented_data = []
        
        # Keep originals + add noisy versions
        for sample in data:
            # Original
            augmented_data.append(sample)
            
            # Noisy version
            noisy = sample.copy()
            noisy['text'] = self.add_pdf_noise(sample['text'], noise_level)
            noisy['id'] = f"{sample['id']}_noisy"
            augmented_data.append(noisy)
        
        with open(output_path, 'w') as f:
            json.dump(augmented_data, f, indent=2)
        
        print(f"Augmented samples: {len(augmented_data):,}")
        print(f"Saved to: {output_path}")


# ============================================================================
# PHASE 6: COMPREHENSIVE EVALUATION
# ============================================================================

class ESGEvaluator:
    """
    Comprehensive evaluation for all 11 layers
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_esg_filter(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Evaluate Layer 4: ESG Candidate Filter
        
        Critical metrics:
        - Precision (avoid processing non-ESG)
        - Recall (don't miss ESG content)
        - Speed improvement estimate
        """
        print("\n" + "="*70)
        print("LAYER 4: ESG CANDIDATE FILTER EVALUATION")
        print("="*70)
        
        true_labels = [s.get('is_esg_candidate', True) for s in ground_truth]
        pred_labels = [s.get('predicted_is_esg', True) for s in predictions]
        
        # Binary metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary'
        )
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # False positive rate (processing non-ESG text)
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if not t and not p)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if not t and p)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': fpr,
            'estimated_speedup': 1 / (1 - (1 - recall) * 0.5)  # Rough estimate
        }
        
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"Precision:     {precision:.4f} (avoid false processing)")
        print(f"Recall:        {recall:.4f} (catch all ESG content)")
        print(f"F1 Score:      {f1:.4f}")
        print(f"False Pos Rate:{fpr:.4f}")
        print(f"Est. Speedup:  {results['estimated_speedup']:.2f}x")
        
        return results
    
    def evaluate_ner(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """Evaluate Layer 5: NER Model"""
        from seqeval.metrics import (
            classification_report, f1_score, precision_score, recall_score
        )
        
        print("\n" + "="*70)
        print("LAYER 5: NER MODEL EVALUATION")
        print("="*70)
        
        true_labels = [sample['tags'] for sample in ground_truth]
        pred_labels = [sample['predicted_tags'] for sample in predictions]
        
        results = {
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
        }
        
        report = classification_report(true_labels, pred_labels, output_dict=True)
        results['per_label'] = report
        
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        
        # Show top/bottom performing entities
        entity_f1s = {k: v['f1-score'] for k, v in report.items() 
                     if isinstance(v, dict) and k.startswith('B-')}
        
        if entity_f1s:
            print("\nBest performing entities:")
            for entity, f1 in sorted(entity_f1s.items(), key=lambda x: -x[1])[:5]:
                print(f"  {entity}: {f1:.4f}")
            
            print("\nWorst performing entities:")
            for entity, f1 in sorted(entity_f1s.items(), key=lambda x: x[1])[:5]:
                print(f"  {entity}: {f1:.4f}")
        
        return results
    
    def evaluate_classification(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """Evaluate Layer 6: Metric Classifier"""
        
        print("\n" + "="*70)
        print("LAYER 6: METRIC CLASSIFIER EVALUATION")
        print("="*70)
        
        true_labels = [s['label'] for s in ground_truth]
        pred_labels = [s['predicted_label'] for s in predictions]
        
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'per_class': report,
            'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist()
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Show per-class F1
        print("\nPer-class F1 scores (top 10):")
        class_f1s = {k: v['f1-score'] for k, v in report.items() 
                    if isinstance(v, dict) and k not in ['accuracy', 'macro avg', 'weighted avg']}
        
        for label, f1 in sorted(class_f1s.items(), key=lambda x: -x[1])[:10]:
            print(f"  {label}: {f1:.4f}")
        
        return results
    
    def evaluate_validation_layer(
        self,
        validation_examples: List[Dict]
    ) -> Dict:
        """
        Evaluate Layer 11: Validation Layer
        
        Tests business rules and range checking
        """
        print("\n" + "="*70)
        print("LAYER 11: VALIDATION LAYER EVALUATION")
        print("="*70)
        
        total = len(validation_examples)
        
        # Group by test type
        by_type = defaultdict(list)
        for ex in validation_examples:
            by_type[ex['test_type']].append(ex)
        
        results = {}
        
        for test_type, examples in by_type.items():
            # Check if validation correctly identified valid/invalid
            correct = sum(1 for ex in examples if ex['is_valid'] == (not ex['issues']))
            accuracy = correct / len(examples) if examples else 0
            
            results[test_type] = {
                'count': len(examples),
                'accuracy': accuracy
            }
            
            print(f"\n{test_type}:")
            print(f"  Samples: {len(examples)}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate_end_to_end(
        self,
        extracted_metrics: List[Dict],
        ground_truth: List[Dict],
        value_tolerance: float = 0.01
    ) -> Dict:
        """
        Evaluate complete end-to-end pipeline (all 11 layers)
        
        Checks:
        - Correct metric identified (Layers 4-6)
        - Correct value extracted (Layers 7-8)
        - Correct unit extracted (Layer 9)
        - Validation passed (Layer 11)
        """
        print("\n" + "="*70)
        print("END-TO-END PIPELINE EVALUATION (ALL 11 LAYERS)")
        print("="*70)
        
        total = len(ground_truth)
        
        correct_metric = 0
        correct_value = 0
        correct_unit = 0
        passed_validation = 0
        fully_correct = 0
        
        for gt in ground_truth:
            pred = next((p for p in extracted_metrics if p['id'] == gt['id']), None)
            
            if not pred:
                continue
            
            # Layer 4-6: Metric identification
            if pred.get('normalized_metric') == gt.get('normalized_metric'):
                correct_metric += 1
                
                # Layer 7-8: Value extraction
                if gt.get('value') is not None:
                    value_diff = abs(pred.get('value', 0) - gt['value'])
                    if value_diff < value_tolerance:
                        correct_value += 1
                
                # Layer 9: Unit normalization
                if pred.get('unit') == gt.get('unit'):
                    correct_unit += 1
                
                # Layer 11: Validation
                if pred.get('validation_status') == 'VALID':
                    passed_validation += 1
                
                # Fully correct
                if (pred.get('normalized_metric') == gt.get('normalized_metric') and
                    abs(pred.get('value', 0) - gt.get('value', 0)) < value_tolerance and
                    pred.get('unit') == gt.get('unit') and
                    pred.get('validation_status') == 'VALID'):
                    fully_correct += 1
        
        results = {
            'total_samples': total,
            'metric_accuracy': correct_metric / total if total > 0 else 0,
            'value_accuracy': correct_value / total if total > 0 else 0,
            'unit_accuracy': correct_unit / total if total > 0 else 0,
            'validation_pass_rate': passed_validation / total if total > 0 else 0,
            'end_to_end_accuracy': fully_correct / total if total > 0 else 0
        }
        
        print(f"Total samples:          {total}")
        print(f"\nPer-Layer Performance:")
        print(f"  Layers 4-6 (Metric):  {results['metric_accuracy']:.2%}")
        print(f"  Layers 7-8 (Value):   {results['value_accuracy']:.2%}")
        print(f"  Layer 9 (Unit):       {results['unit_accuracy']:.2%}")
        print(f"  Layer 11 (Validation):{results['validation_pass_rate']:.2%}")
        print(f"\nEnd-to-End (All 11):    {results['end_to_end_accuracy']:.2%}")
        
        return results
    
    def analyze_errors(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Detailed error analysis across all layers
        """
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)
        
        errors = {
            'layer4_false_negatives': [],  # Missed ESG content
            'layer4_false_positives': [],  # Processed non-ESG
            'layer5_missed_entities': [],  # NER failures
            'layer6_wrong_classification': [],  # Classifier errors
            'layer7_8_value_errors': [],  # Value extraction failures
            'layer9_unit_errors': [],  # Unit normalization issues
            'layer11_validation_failures': []  # Validation errors
        }
        
        for gt in ground_truth:
            pred = next((p for p in predictions if p['id'] == gt['id']), None)
            
            if not pred:
                errors['layer5_missed_entities'].append(gt)
                continue
            
            # Layer 4 errors
            if gt.get('is_esg_candidate') and not pred.get('predicted_is_esg'):
                errors['layer4_false_negatives'].append(gt)
            elif not gt.get('is_esg_candidate') and pred.get('predicted_is_esg'):
                errors['layer4_false_positives'].append(gt)
            
            # Layer 6 errors
            if pred.get('normalized_metric') != gt.get('normalized_metric'):
                errors['layer6_wrong_classification'].append({
                    'true': gt['normalized_metric'],
                    'predicted': pred['normalized_metric'],
                    'text': gt.get('text', '')[:100]
                })
            
            # Layer 7-8 errors
            if abs(pred.get('value', 0) - gt.get('value', 0)) > 0.01:
                errors['layer7_8_value_errors'].append({
                    'true': gt.get('value'),
                    'predicted': pred.get('value'),
                    'metric': gt['normalized_metric']
                })
            
            # Layer 9 errors
            if pred.get('unit') != gt.get('unit'):
                errors['layer9_unit_errors'].append({
                    'true': gt.get('unit'),
                    'predicted': pred.get('unit'),
                    'metric': gt['normalized_metric']
                })
            
            # Layer 11 errors
            if pred.get('validation_status') != 'VALID' and gt.get('is_valid', True):
                errors['layer11_validation_failures'].append({
                    'metric': pred['normalized_metric'],
                    'value': pred['value'],
                    'issues': pred.get('validation_issues', [])
                })
        
        # Summarize
        print("\nError counts by layer:")
        for layer, error_list in errors.items():
            if error_list:
                print(f"  {layer}: {len(error_list)}")
        
        # Most common issues
        if errors['layer6_wrong_classification']:
            confusion_pairs = Counter(
                (e['true'], e['predicted']) 
                for e in errors['layer6_wrong_classification']
            )
            print("\nMost confused metric pairs (Layer 6):")
            for (true, pred), count in confusion_pairs.most_common(5):
                print(f"  {true} → {pred}: {count} times")
        
        return errors
    
    def evaluate_confidence_calibration(
        self,
        predictions: List[Dict]
    ) -> Dict:
        """
        Evaluate Layer 10: Confidence Scoring
        
        Check if confidence scores are well-calibrated
        """
        print("\n" + "="*70)
        print("LAYER 10: CONFIDENCE CALIBRATION")
        print("="*70)
        
        bins = np.linspace(0, 1, 11)
        bin_accuracy = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_low = bins[i]
            bin_high = bins[i+1]
            
            bin_preds = [
                p for p in predictions
                if bin_low <= p.get('confidence_score', 0) < bin_high
            ]
            
            if not bin_preds:
                continue
            
            correct = sum(1 for p in bin_preds if p.get('is_correct', False))
            accuracy = correct / len(bin_preds)
            
            bin_accuracy.append(accuracy)
            bin_counts.append(len(bin_preds))
            
            print(f"Confidence {bin_low:.1f}-{bin_high:.1f}: "
                  f"Accuracy={accuracy:.2%} (n={len(bin_preds)})")
        
        # Calculate calibration error
        calibration_error = 0
        for i, (acc, count) in enumerate(zip(bin_accuracy, bin_counts)):
            expected_conf = (bins[i] + bins[i+1]) / 2
            calibration_error += abs(acc - expected_conf) * count
        
        calibration_error /= len(predictions)
        
        print(f"\nExpected Calibration Error: {calibration_error:.4f}")
        print("(Lower is better, <0.05 is well-calibrated)")
        
        return {
            'bin_accuracy': bin_accuracy,
            'bin_counts': bin_counts,
            'calibration_error': calibration_error
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive evaluation and robustness testing"""
    
    print("\n" + "="*70)
    print("ESG EXTRACTION: EVALUATION & ROBUSTNESS (ALL 11 LAYERS)")
    print("="*70)
    
    # 1. Add PDF noise to datasets
    print("\n" + "="*70)
    print("STEP 1: Adding Real-World PDF Noise")
    print("="*70)
    
    bridge = SyntheticToRealBridge()
    
    data_dir = Path('./processed_data')
    if data_dir.exists():
        for dataset in ['ner_train.json', 'classification_train.json']:
            input_path = data_dir / dataset
            output_path = data_dir / f'augmented_{dataset}'
            
            if input_path.exists():
                bridge.augment_dataset(
                    str(input_path),
                    str(output_path),
                    noise_level=0.3
                )
    
    # 2. Print comprehensive evaluation guide
    print("\n" + "="*70)
    print("EVALUATION GUIDE")
    print("="*70)
    
    evaluation_guide = """
To evaluate the complete 11-layer pipeline:

1. PREPARE TEST DATA
   Load: ./processed_data/*_test.json
   Ensure: Ground truth labels for all layers

2. RUN INFERENCE
   - Load trained models (ESG filter, NER, Classifier)
   - Process test samples through complete pipeline
   - Save predictions with confidence scores

3. EVALUATE EACH LAYER
   evaluator = ESGEvaluator()
   
   # Layer 4: ESG Filter
   filter_results = evaluator.evaluate_esg_filter(predictions, ground_truth)
   
   # Layer 5: NER
   ner_results = evaluator.evaluate_ner(predictions, ground_truth)
   
   # Layer 6: Classifier
   class_results = evaluator.evaluate_classification(predictions, ground_truth)
   
   # Layer 11: Validation
   val_results = evaluator.evaluate_validation_layer(validation_examples)
   
   # Layers 10: Confidence
   conf_results = evaluator.evaluate_confidence_calibration(predictions)
   
   # End-to-End (All layers)
   e2e_results = evaluator.evaluate_end_to_end(predictions, ground_truth)
   
   # Error Analysis
   errors = evaluator.analyze_errors(predictions, ground_truth)

4. TARGET METRICS
   Layer 4 (ESG Filter):
     - Recall: >0.95 (don't miss ESG content)
     - Precision: >0.90 (avoid processing junk)
   
   Layer 5 (NER):
     - F1 Score: >0.92
   
   Layer 6 (Classifier):
     - Accuracy: >0.95
   
   Layers 7-9 (Value Extraction):
     - Exact match: >0.85
   
   Layer 11 (Validation):
     - Correct flagging: >0.95
   
   End-to-End (All 11 layers):
     - Full accuracy: >0.80

5. ITERATE
   - Identify weakest layer
   - Add targeted training examples
   - Retrain affected models
   - Re-evaluate
    """
    
    print(evaluation_guide)
    
    print("\n" + "="*70)
    print("✓ ROBUSTNESS & EVALUATION SETUP COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
