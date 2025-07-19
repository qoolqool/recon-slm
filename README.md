# Report Reconciliation SLM Training

This project trains a Small Language Model (SLM) for automated report reconciliation tasks.

## Quick Start

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Generate training data and train the model:**
   ```bash
   docker-compose run reconciliation-trainer
   ```

3. **Or run specific commands:**
   ```bash
   # Generate data only
   docker-compose run reconciliation-trainer python run_reconciliation_training.py --num-records 2000

   # Train with custom settings
   docker-compose run reconciliation-trainer python run_reconciliation_training.py --epochs 5 --model-name "distilgpt2"

   # Demo existing model
   docker-compose run reconciliation-trainer python run_reconciliation_training.py --demo-only
   ```

4. **Start Jupyter notebook for development:**
   ```bash
   docker-compose up jupyter
   ```
   Then open http://localhost:8888 in your browser.

## Project Structure

- `data/` - Training data and examples
- `models/` - Trained model outputs
- `logs/` - Training logs
- `results/` - Evaluation results
- `notebooks/` - Jupyter notebooks for exploration

## Model Capabilities

The trained model can:
- Identify perfect matches between systems
- Detect amount discrepancies
- Identify date mismatches
- Detect missing transactions
- Provide explanations for reconciliation decisions

## Example Usage

```python
from reconciliation_trainer import ReconciliationTrainer

trainer = ReconciliationTrainer()
# Load your trained model
trainer.model = trainer.model.__class__.from_pretrained("models/reconciliation_model")
trainer.tokenizer = trainer.tokenizer.__class__.from_pretrained("models/reconciliation_model")

# Analyze transactions
analysis = trainer.generate_reconciliation_report(system_a_record, system_b_record)
print(analysis)
```
