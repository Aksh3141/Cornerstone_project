"""
Evaluation Script for Violence Detection Model
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class ModelEvaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def evaluate(self):
        """Run complete evaluation"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("Running evaluation...")
        with torch.no_grad():
            for frames, labels in tqdm(self.test_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(frames)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of violence class
        
        # Calculate metrics
        results = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Generate visualizations
        self._plot_confusion_matrix(all_labels, all_preds)
        self._plot_roc_curve(all_labels, all_probs)
        
        return results
    
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate all evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        
        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None)
        recall_per_class = recall_score(labels, preds, average=None)
        f1_per_class = f1_score(labels, preds, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC
        auc = roc_auc_score(labels, probs)
        
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'auc_roc': float(auc)
            },
            'per_class': {
                'non_violence': {
                    'precision': float(precision_per_class[0]),
                    'recall': float(recall_per_class[0]),
                    'f1_score': float(f1_per_class[0])
                },
                'violence': {
                    'precision': float(precision_per_class[1]),
                    'recall': float(recall_per_class[1]),
                    'f1_score': float(f1_per_class[1])
                }
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """Print formatted results"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        overall = results['overall']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:    {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
        print(f"  Precision:   {overall['precision']:.4f}")
        print(f"  Recall:      {overall['recall']:.4f}")
        print(f"  F1-Score:    {overall['f1_score']:.4f}")
        print(f"  Specificity: {overall['specificity']:.4f}")
        print(f"  AUC-ROC:     {overall['auc_roc']:.4f}")
        
        cm = results['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm['true_negatives']:4d}")
        print(f"  False Positives: {cm['false_positives']:4d}")
        print(f"  False Negatives: {cm['false_negatives']:4d}")
        print(f"  True Positives:  {cm['true_positives']:4d}")
        
        per_class = results['per_class']
        print(f"\nPer-Class Metrics:")
        print(f"  Non-Violence:")
        print(f"    Precision: {per_class['non_violence']['precision']:.4f}")
        print(f"    Recall:    {per_class['non_violence']['recall']:.4f}")
        print(f"    F1-Score:  {per_class['non_violence']['f1_score']:.4f}")
        print(f"  Violence:")
        print(f"    Precision: {per_class['violence']['precision']:.4f}")
        print(f"    Recall:    {per_class['violence']['recall']:.4f}")
        print(f"    F1-Score:  {per_class['violence']['f1_score']:.4f}")
        
        print("=" * 60)
    
    def _plot_confusion_matrix(self, labels, preds):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved to confusion_matrix.png")
        plt.close()
    
    def _plot_roc_curve(self, labels, probs):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved to roc_curve.png")
        plt.close()


# Usage example
if __name__ == "__main__":
    from violence_model import ViolenceDetectionModel
    from rwf_dataset_loader import get_dataloaders
    
    # Load test data
    _, test_loader = get_dataloaders(
        dataset_info_path='./processed_rwf2000/dataset_info.json',
        batch_size=8,
        num_workers=4
    )
    
    # Load trained model
    model = ViolenceDetectionModel(num_classes=2)
    checkpoint = torch.load('./checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(model, test_loader, device=device)
    results = evaluator.evaluate()
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to evaluation_results.json")
