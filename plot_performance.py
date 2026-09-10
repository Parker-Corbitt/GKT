import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(log_files, output_dir):
    """
    Plots training and validation metrics from multiple log files.
    Labels are autopopulated from the 'memory' column in the CSV.
    """
    metrics_to_plot = {
        'loss': ('train_loss', 'val_loss', 'Loss'),
        'auc': ('train_auc', 'val_auc', 'AUC'),
        'acc': ('train_acc', 'val_acc', 'Accuracy')
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric_key, (train_col, val_col, title) in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))

        for log_file in log_files:
            try:
                df = pd.read_csv(log_file)

                # Autopopulate label from the first row's 'memory' column
                label = df['memory'].iloc[0] if 'memory' in df.columns else os.path.basename(os.path.dirname(log_file))

                # Plot Training
                plt.plot(df['epoch'], df[train_col], label=f'{label} (Train)', linestyle='--', alpha=0.7)
                # Plot Validation
                plt.plot(df['epoch'], df[val_col], label=f'{label} (Val)', linewidth=2)

            except Exception as e:
                print(f"Error reading {log_file}: {e}")

        plt.title(f'Training and Validation {title} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(output_dir, f'{metric_key}_plot.png')
        plt.savefig(save_path)
        print(f"Saved {title} plot to {save_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GKT performance metrics from CSV logs.')
    parser.add_argument('--logs', nargs='+', required=True, help='Paths to the performance_log.csv files.')
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory to save the generated plots.')

    args = parser.parse_args()

    plot_metrics(args.logs, args.output_dir)
    print("\nAll plots have been generated successfully.")
