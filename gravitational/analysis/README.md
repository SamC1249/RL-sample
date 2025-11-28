# Training Analysis Tools

This folder contains tools for analyzing DQN training results, similar to TensorBoard functionality.

## Files

### `visualize_training.py`
Comprehensive training visualization tool that analyzes checkpoints and generates detailed reports.

**Usage:**
```bash
# Activate virtual environment first
cd C:\Users\chens\AI and Coding\Classes\AM158\RL-sample
.\venv\Scripts\Activate.ps1

# Run analysis (uses default checkpoint directory)
python gravitational/analysis/visualize_training.py

# Or specify custom checkpoint directory
python gravitational/analysis/visualize_training.py path/to/checkpoints
```

**Generated Outputs:**
- `training_dashboard.png` - Main dashboard with all metrics
- `weight_evolution.png` - Q-network weight changes over time
- `problem_diagnosis.png` - Detailed problem analysis
- `analysis_report.txt` - Text summary report

All outputs are saved to: `checkpoints/analysis/results/`

### `FINDINGS.md`
Detailed analysis of why the DQN agent has low rewards, including:
- Root cause analysis
- Code-level explanations
- Specific fixes with code examples
- Expected results after fixes

## Key Findings

The agent's poor performance (0% success rate, -3,464 average reward) is caused by:

1. **Critical Bug**: Terminal condition detection never identifies success
2. **Reward Structure**: Gravity costs dominate, no positive shaping
3. **Limited State**: Agent doesn't know distance/direction to target or black hole

See `FINDINGS.md` for complete details and fixes.

## Visualizations

The tool generates TensorBoard-style visualizations showing:
- Training statistics summary
- Weight magnitude evolution
- Epsilon decay curve
- Reward analysis
- Episode outcome distribution
- Learning progress gauge
- Detailed recommendations

## Requirements

All required packages are in the main `requirements.txt`. Key dependencies:
- matplotlib
- seaborn
- torch
- numpy

## Quick Start

1. Ensure you've trained a model (checkpoints exist in `gravitational/checkpoints/dqn/`)
2. Run the visualization tool (see Usage above)
3. Check generated images in `checkpoints/analysis/results/`
4. Read `FINDINGS.md` for detailed analysis
5. Implement recommended fixes
6. Retrain and compare results

