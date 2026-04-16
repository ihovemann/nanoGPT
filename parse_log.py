import re
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
from datetime import timedelta

def parse_log_file(log_file_path):
    """
    Parse a nanoGPT training log file.
    
    Returns:
        dict: Contains 'steps', 'train_losses', 'val_losses', 'iter_losses', 'config', 'timing'
    """
    steps = []
    train_losses = []
    val_losses = []

    iter_nums = []
    iter_losses = []

    # Configuration parameters
    config = {
        'learning_rate': None,
        'n_layer': None,
        'n_embd': None,
        'block_size': None,
        'dropout': None,
        'max_iters': None,
    }

    # Timing information
    timing_info = {
        'start_time': None,
        'end_time': None,
        'total_time_ms': 0,
    }

    print(f"\nParsing: {log_file_path}")

    # Try different encodings, including UTF-16
    for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(log_file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"  Successfully read with encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        print(f"  Warning: Could not decode file properly")
        return {
            'steps': [], 'train_losses': [], 'val_losses': [],
            'iter_nums': [], 'iter_losses': [], 'config': config, 'timing': timing_info
        }

    iter_times = []

    for line_num, line in enumerate(lines, 1):
        # Parse configuration parameters
        lr_match = re.search(r'learning_rate\s*=\s*([\d.e-]+)', line)
        if lr_match:
            config['learning_rate'] = float(lr_match.group(1))

        layer_match = re.search(r'n_layer\s*=\s*(\d+)', line)
        if layer_match:
            config['n_layer'] = int(layer_match.group(1))

        embd_match = re.search(r'n_embd\s*=\s*(\d+)', line)
        if embd_match:
            config['n_embd'] = int(embd_match.group(1))

        block_match = re.search(r'block_size\s*=\s*(\d+)', line)
        if block_match:
            config['block_size'] = int(block_match.group(1))

        dropout_match = re.search(r'dropout\s*=\s*([\d.]+)', line)
        if dropout_match:
            config['dropout'] = float(dropout_match.group(1))

        iters_match = re.search(r'max_iters\s*=\s*(\d+)', line)
        if iters_match:
            config['max_iters'] = int(iters_match.group(1))

        # Match lines like: "step 0: train loss 4.2874, val loss 4.2823"
        step_match = re.search(r'step\s+(\d+):\s+train\s+loss\s+([\d.]+),\s+val\s+loss\s+([\d.]+)', line)
        if step_match:
            step = int(step_match.group(1))
            train_loss = float(step_match.group(2))
            val_loss = float(step_match.group(3))
            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if len(steps) <= 3:  # Print first 3 matches
                print(f"  Line {line_num}: step {step}, train={train_loss}, val={val_loss}")

        # Match lines like: "iter 10: loss 3.1464, time 619.09ms, mfu 0.60%"
        iter_match = re.search(r'iter\s+(\d+):\s+loss\s+([\d.]+),\s+time\s+([\d.]+)ms', line)
        if iter_match:
            iter_num = int(iter_match.group(1))
            iter_loss = float(iter_match.group(2))
            iter_time = float(iter_match.group(3))
            iter_nums.append(iter_num)
            iter_losses.append(iter_loss)
            iter_times.append(iter_time)

    # Calculate total training time from iter times
    if iter_times:
        timing_info['total_time_ms'] = sum(iter_times)

    print(f"  Total steps found: {len(steps)}")
    print(f"  Total iters found: {len(iter_nums)}")

    if len(steps) > 0:
        print(f"  Step range: {min(steps)} to {max(steps)}")
        print(f"  Train loss range: {min(train_losses):.4f} to {max(train_losses):.4f}")
        print(f"  Val loss range: {min(val_losses):.4f} to {max(val_losses):.4f}")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Final val loss: {val_losses[-1]:.4f}")

    if timing_info['total_time_ms'] > 0:
        total_min = timing_info['total_time_ms'] / 1000 / 60
        print(f"  Total training time: {total_min:.2f} minutes")

    return {
        'steps': steps,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'iter_nums': iter_nums,
        'iter_losses': iter_losses,
        'config': config,
        'timing': timing_info
    }

def create_summary_table(all_data, run_names, output_dir='plots'):
    """
    Create a summary table with experiment results.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    for data, name in zip(all_data, run_names):
        if len(data['steps']) == 0:
            continue

        config = data['config']

        # Get final losses
        final_train_loss = data['train_losses'][-1] if data['train_losses'] else None
        final_val_loss = data['val_losses'][-1] if data['val_losses'] else None

        # Calculate time in minutes
        time_min = data['timing']['total_time_ms'] / 1000 / 60 if data['timing']['total_time_ms'] > 0 else None

        summary_data.append({
            'Experiment': name,
            'LR': config['learning_rate'],
            'Layers': config['n_layer'],
            'Embd': config['n_embd'],
            'Block': config['block_size'],
            'Dropout': config['dropout'],
            'Iters': config['max_iters'],
            'Train Loss': final_train_loss,
            'Val Loss': final_val_loss,
            'Time (min)': time_min
        })

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Format the dataframe for nice display
    pd.options.display.float_format = '{:.4f}'.format

    # Print to console
    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'experiment_summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✓ Saved summary table to {csv_path}")

    # Save to LaTeX (useful for papers/reports)
    latex_path = os.path.join(output_dir, 'experiment_summary.tex')
    df.to_latex(latex_path, index=False, float_format='%.4f')
    print(f"✓ Saved LaTeX table to {latex_path}")

    # Save to Markdown
    md_path = os.path.join(output_dir, 'experiment_summary.md')
    with open(md_path, 'w') as f:
        f.write("# Experiment Summary\n\n")
        f.write(df.to_markdown(index=False, floatfmt='.4f'))
    print(f"✓ Saved Markdown table to {md_path}")

    return df

def plot_individual_run(data, run_name, output_dir='plots'):
    """
    Create separate training and validation plots for a single run.
    """
    if len(data['steps']) == 0:
        print(f"  ⚠️  No data to plot for {run_name}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with 2 subplots (training and validation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training Loss
    ax1.plot(data['steps'], data['train_losses'], 
            marker='o', color='#2E86AB', linewidth=2, markersize=5)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title(f'{run_name} - Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax2.plot(data['steps'], data['val_losses'], 
            marker='s', color='#A23B72', linewidth=2, markersize=5)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'{run_name} - Validation Loss', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'{run_name}_losses.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  ✓ Saved individual plot to {output_path}")
    plt.close()

def plot_combined_losses(all_data, run_names, output_dir='plots'):
    """
    Plot training and validation losses from multiple runs in combined plots.
    """

    # Check if we have any data
    if not any(len(data['steps']) > 0 for data in all_data):
        print("\n⚠️  WARNING: No step data found in any log files!")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Combined Training Loss
    plt.figure(figsize=(10, 6))
    for data, name in zip(all_data, run_names):
        if len(data['steps']) > 0:
            plt.plot(data['steps'], data['train_losses'], 
                    marker='o', label=name, linewidth=2, markersize=4)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss vs. Step - All Runs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'combined_training_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ Saved combined training loss plot to {output_path}")
    plt.close()

    # Plot 2: Combined Validation Loss
    plt.figure(figsize=(10, 6))
    for data, name in zip(all_data, run_names):
        if len(data['steps']) > 0:
            plt.plot(data['steps'], data['val_losses'], 
                    marker='s', label=name, linewidth=2, markersize=4)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss vs. Step - All Runs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'combined_validation_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved combined validation loss plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    logs_dir = 'logs'
    plots_dir = 'plots'

    # Get all .log files from the logs directory
    log_files = glob.glob(os.path.join(logs_dir, '*.log'))
    log_files.sort()

    if not log_files:
        print(f"No .log files found in {logs_dir}")
        exit(1)

    print(f"Found {len(log_files)} log file(s):")
    for log_file in log_files:
        print(f"  - {log_file}")

    # Generate run names from filenames
    run_names = [os.path.splitext(os.path.basename(f))[0] for f in log_files]

    # Parse all log files
    print("\n" + "="*80)
    print("PARSING LOG FILES")
    print("="*80)
    all_data = []
    for log_file in log_files:
        data = parse_log_file(log_file)
        all_data.append(data)

    # Create summary table
    create_summary_table(all_data, run_names, output_dir=plots_dir)

    # Create individual plots for each run
    print("\n" + "="*80)
    print("CREATING INDIVIDUAL PLOTS")
    print("="*80)
    for data, name in zip(all_data, run_names):
        print(f"\nPlotting {name}...")
        plot_individual_run(data, name, output_dir=plots_dir)

    # Create combined plots
    print("\n" + "="*80)
    print("CREATING COMBINED PLOTS")
    print("="*80)
    plot_combined_losses(all_data, run_names, output_dir=plots_dir)

    print("\n" + "="*80)
    print("DONE! All plots and summary table saved to 'plots/' folder")
    print("="*80)