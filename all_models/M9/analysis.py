import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class GATAnalyzer:
    def __init__(self, results_path):
        self.df = pd.read_csv(results_path)
        self.setup_plotting()

    def setup_plotting(self):
        """Set consistent plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def parameter_analysis(self):
        """Analyze impact of each parameter"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Delta Impact
        delta_groups = self.df.groupby('delta')['auc'].agg(['mean', 'std', 'count'])
        axes[0, 0].bar(delta_groups.index, delta_groups['mean'],
                       yerr=delta_groups['std'], capsize=5)
        axes[0, 0].set_xlabel('Delta')
        axes[0, 0].set_ylabel('Average AUC')
        axes[0, 0].set_title('Impact of Delta Parameter')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cutoff Impact
        cutoff_groups = self.df.groupby('cutoff')['auc'].agg(['mean', 'std'])
        axes[0, 1].bar(cutoff_groups.index, cutoff_groups['mean'],
                       yerr=cutoff_groups['std'], capsize=5)
        axes[0, 1].set_xlabel('Cutoff')
        axes[0, 1].set_ylabel('Average AUC')
        axes[0, 1].set_title('Impact of Cutoff Parameter')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Minedge Impact
        minedge_groups = self.df.groupby('minedge')['auc'].agg(['mean', 'std'])
        axes[1, 0].bar(minedge_groups.index, minedge_groups['mean'],
                       yerr=minedge_groups['std'], capsize=5)
        axes[1, 0].set_xlabel('Minedge')
        axes[1, 0].set_ylabel('Average AUC')
        axes[1, 0].set_title('Impact of Minedge Parameter')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 3D Parameter Interaction
        pivot = self.df.pivot_table(
            values='auc',
            index=['delta', 'minedge'],
            columns='cutoff',
            aggfunc='mean'
        )
        im = axes[1, 1].imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xlabel('Cutoff')
        axes[1, 1].set_ylabel('(Delta, Minedge)')
        axes[1, 1].set_title('Parameter Interaction Heatmap')
        plt.colorbar(im, ax=axes[1, 1])

        # Set tick labels
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_xticklabels(pivot.columns)
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels([f"({d},{m})" for d, m in pivot.index])

        plt.tight_layout()
        plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        self.summary_statistics()
        self.parameter_analysis()

        # Save detailed analysis
        report = {
            'summary': {
                'total_datasets': len(self.df),
                'mean_auc': float(self.df['auc'].mean()),
                'max_auc': float(self.df['auc'].max()),
                'min_auc': float(self.df['auc'].min()),
                'std_auc': float(self.df['auc'].std())
            },
            'best_configurations': self.df.nlargest(5, 'auc').to_dict('records'),
            'parameter_impact': {
                'delta': self.df.groupby('delta')['auc'].mean().to_dict(),
                'cutoff': self.df.groupby('cutoff')['auc'].mean().to_dict(),
                'minedge': self.df.groupby('minedge')['auc'].mean().to_dict()
            }
        }

        with open('analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    analyzer = GATAnalyzer('results/results_final_20251216_043721.csv')
