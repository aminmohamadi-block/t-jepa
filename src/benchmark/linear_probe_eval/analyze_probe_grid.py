#!/usr/bin/env python3
"""
Analyze results from linear probe grid search
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def parse_slurm_output(output_file):
    """Parse metrics from SLURM output file"""
    if not os.path.exists(output_file):
        return None
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Extract parameters
        lr_match = re.search(r'Learning Rate: ([\d.e-]+)', content)
        wd_match = re.search(r'Weight Decay: ([\d.e-]+)', content)
        
        # Extract final validation loss (handle N/A case)
        val_loss_match = re.search(r'Final validation loss: ([\d.e-]+|N/A)', content)
        
        # Extract test results from PyTorch Lightning table format
        test_loss_match = re.search(r'higgs_test_loss\s+([\d.e-]+)', content)
        test_acc_match = re.search(r'higgs_test_accuracy\s+([\d.e-]+)', content)
        test_precision_match = re.search(r'higgs_test_precision\s+([\d.e-]+)', content)
        test_recall_match = re.search(r'higgs_test_recall\s+([\d.e-]+)', content)
        test_f1_match = re.search(r'higgs_test_f1-score\s+([\d.e-]+)', content)
        test_roc_auc_match = re.search(r'higgs_test_roc_auc\s+([\d.e-]+)', content)
        
        # Check success/failure
        success = "✅ Linear probe optimization SUCCESS" in content
        
        if lr_match and wd_match:
            # Handle val_loss (could be N/A)
            val_loss = None
            if val_loss_match and val_loss_match.group(1) != 'N/A':
                val_loss = float(val_loss_match.group(1))
            
            result = {
                'learning_rate': float(lr_match.group(1)),
                'weight_decay': float(wd_match.group(1)),
                'val_loss': val_loss,
                'test_loss': float(test_loss_match.group(1)) if test_loss_match else None,
                'test_accuracy': float(test_acc_match.group(1)) if test_acc_match else None,
                'test_precision': float(test_precision_match.group(1)) if test_precision_match else None,
                'test_recall': float(test_recall_match.group(1)) if test_recall_match else None,
                'test_f1_score': float(test_f1_match.group(1)) if test_f1_match else None,
                'test_roc_auc': float(test_roc_auc_match.group(1)) if test_roc_auc_match else None,
                'success': success,
                'output_file': output_file
            }
            return result
            
    except Exception as e:
        print(f"Error parsing {output_file}: {e}")
    
    return None


def analyze_grid_results(results_dir):
    """Analyze all results from grid search"""
    
    print(f"Analyzing results in: {results_dir}")
    
    # Find all output files
    output_files = list(Path(results_dir).glob("probe_grid_*.out"))
    
    if not output_files:
        print("No output files found!")
        return None
    
    print(f"Found {len(output_files)} output files")
    
    # Parse all results
    results = []
    for output_file in output_files:
        result = parse_slurm_output(output_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"Successfully parsed {len(df)} results")
    
    return df


def create_visualizations(df, results_dir):
    """Create visualization plots"""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Heatmap of validation loss
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        pivot_val = df.pivot(index='weight_decay', columns='learning_rate', values='val_loss')
        sns.heatmap(pivot_val, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,0])
        axes[0,0].set_title('Validation Loss Heatmap')
        axes[0,0].set_xlabel('Learning Rate')
        axes[0,0].set_ylabel('Weight Decay')
    
    # 2. Heatmap of test accuracy
    if 'test_accuracy' in df.columns and df['test_accuracy'].notna().any():
        pivot_acc = df.pivot(index='weight_decay', columns='learning_rate', values='test_accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0,1])
        axes[0,1].set_title('Test Accuracy Heatmap')
        axes[0,1].set_xlabel('Learning Rate')
        axes[0,1].set_ylabel('Weight Decay')
    
    # 3. Success rate by learning rate
    if 'success' in df.columns:
        success_by_lr = df.groupby('learning_rate')['success'].mean()
        success_by_lr.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Success Rate by Learning Rate')
        axes[1,0].set_xlabel('Learning Rate')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Success rate by weight decay
    if 'success' in df.columns:
        success_by_wd = df.groupby('weight_decay')['success'].mean()
        success_by_wd.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Success Rate by Weight Decay')
        axes[1,1].set_xlabel('Weight Decay')
        axes[1,1].set_ylabel('Success Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, 'grid_search_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved: {plot_path}")


def print_summary(df):
    """Print summary statistics"""
    
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*60)
    
    # Basic stats
    print(f"Total experiments: {len(df)}")
    
    if 'success' in df.columns:
        success_count = df['success'].sum()
        print(f"Successful experiments: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    
    # Best results
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        best_val_idx = df['val_loss'].idxmin()
        best_val = df.loc[best_val_idx]
        print(f"\nBest Validation Loss: {best_val['val_loss']:.4f}")
        print(f"  Learning Rate: {best_val['learning_rate']}")
        print(f"  Weight Decay: {best_val['weight_decay']}")
        if 'test_accuracy' in best_val:
            print(f"  Test Accuracy: {best_val['test_accuracy']:.4f}")
    
    if 'test_accuracy' in df.columns and df['test_accuracy'].notna().any():
        best_acc_idx = df['test_accuracy'].idxmax()
        best_acc = df.loc[best_acc_idx]
        print(f"\nBest Test Accuracy: {best_acc['test_accuracy']:.4f}")
        print(f"  Learning Rate: {best_acc['learning_rate']}")
        print(f"  Weight Decay: {best_acc['weight_decay']}")
        if 'val_loss' in best_acc and best_acc['val_loss'] is not None:
            print(f"  Validation Loss: {best_acc['val_loss']:.4f}")
        if 'test_precision' in best_acc and best_acc['test_precision'] is not None:
            print(f"  Test Precision: {best_acc['test_precision']:.4f}")
        if 'test_recall' in best_acc and best_acc['test_recall'] is not None:
            print(f"  Test Recall: {best_acc['test_recall']:.4f}")
        if 'test_f1_score' in best_acc and best_acc['test_f1_score'] is not None:
            print(f"  Test F1-Score: {best_acc['test_f1_score']:.4f}")
        if 'test_roc_auc' in best_acc and best_acc['test_roc_auc'] is not None:
            print(f"  Test ROC-AUC: {best_acc['test_roc_auc']:.4f}")
    
    # Parameter ranges that worked
    if 'success' in df.columns:
        successful_df = df[df['success']]
        if len(successful_df) > 0:
            print(f"\nSuccessful Parameter Ranges:")
            print(f"  Learning Rate: {successful_df['learning_rate'].min():.4f} - {successful_df['learning_rate'].max():.4f}")
            print(f"  Weight Decay: {successful_df['weight_decay'].min():.4f} - {successful_df['weight_decay'].max():.4f}")
    
    # Distribution stats
    print(f"\nParameter Grid Coverage:")
    print(f"  Learning Rates tested: {sorted(df['learning_rate'].unique())}")
    print(f"  Weight Decays tested: {sorted(df['weight_decay'].unique())}")
    
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        print(f"\nValidation Loss Statistics:")
        print(f"  Mean: {df['val_loss'].mean():.4f}")
        print(f"  Std: {df['val_loss'].std():.4f}")
        print(f"  Min: {df['val_loss'].min():.4f}")
        print(f"  Max: {df['val_loss'].max():.4f}")


def save_results(df, results_dir):
    """Save results to CSV and JSON"""
    
    # Save detailed CSV
    csv_path = os.path.join(results_dir, 'grid_search_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'total_experiments': len(df),
        'parameter_grid': {
            'learning_rates': sorted(df['learning_rate'].unique().tolist()),
            'weight_decays': sorted(df['weight_decay'].unique().tolist())
        }
    }
    
    if 'success' in df.columns:
        summary['successful_experiments'] = int(df['success'].sum())
        summary['success_rate'] = float(df['success'].mean())
    
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        best_val_idx = df['val_loss'].idxmin()
        best_val = df.loc[best_val_idx]
        summary['best_validation'] = {
            'loss': float(best_val['val_loss']),
            'learning_rate': float(best_val['learning_rate']),
            'weight_decay': float(best_val['weight_decay']),
            'test_accuracy': float(best_val.get('test_accuracy', 0))
        }
    
    if 'test_accuracy' in df.columns and df['test_accuracy'].notna().any():
        best_acc_idx = df['test_accuracy'].idxmax()
        best_acc = df.loc[best_acc_idx]
        summary['best_accuracy'] = {
            'accuracy': float(best_acc['test_accuracy']),
            'learning_rate': float(best_acc['learning_rate']),
            'weight_decay': float(best_acc['weight_decay']),
            'val_loss': float(best_acc['val_loss']) if best_acc.get('val_loss') is not None else None,
            'test_precision': float(best_acc['test_precision']) if best_acc.get('test_precision') is not None else None,
            'test_recall': float(best_acc['test_recall']) if best_acc.get('test_recall') is not None else None,
            'test_f1_score': float(best_acc['test_f1_score']) if best_acc.get('test_f1_score') is not None else None,
            'test_roc_auc': float(best_acc['test_roc_auc']) if best_acc.get('test_roc_auc') is not None else None,
        }
    
    json_path = os.path.join(results_dir, 'grid_search_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze linear probe grid search results")
    parser.add_argument("--results_dir", default="probe_grid_results", 
                       help="Directory containing grid search results")
    parser.add_argument("--no_plots", action="store_true", 
                       help="Skip creating visualization plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return
    
    # Analyze results
    df = analyze_grid_results(args.results_dir)
    
    if df is None:
        print("No results to analyze")
        return
    
    # Print summary
    print_summary(df)
    
    # Save results
    save_results(df, args.results_dir)
    
    # Create visualizations
    if not args.no_plots:
        try:
            create_visualizations(df, args.results_dir)
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("Install matplotlib and seaborn for visualizations")


if __name__ == "__main__":
    main()