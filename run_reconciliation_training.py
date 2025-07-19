#!/usr/bin/env python3
"""
Main runner script for Report Reconciliation SLM Training
"""

import os
import json
import argparse
from pathlib import Path

# Import our custom modules
from data_generator import ReconciliationDataGenerator
from reconciliation_trainer import ReconciliationTrainer

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs', 'results']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)

def generate_data(num_records=1500):
    """Generate training data"""
    print("=" * 50)
    print("GENERATING TRAINING DATA")
    print("=" * 50)

    generator = ReconciliationDataGenerator()
    dataset = generator.generate_full_dataset(num_records)

    # Save data
    dataset['tlf'].to_csv('data/tlf_data.csv', index=False)
    dataset['ilf'].to_csv('data/ilf_data.csv', index=False)

    with open('data/training_examples.json', 'w') as f:
        json.dump(dataset['training_examples'], f, indent=2)

    print(f"✓ Generated {len(dataset['training_examples'])} training examples")
    print(f"✓ TLF records: {len(dataset['tlf'])}")
    print(f"✓ ILF records: {len(dataset['ilf'])}")
    print(f"✓ Data saved to 'data/' directory")

    return dataset

def train_model(dataset, model_name="microsoft/DialoGPT-small", epochs=3):
    """Train the reconciliation model"""
    print("=" * 50)
    print("TRAINING RECONCILIATION MODEL")
    print("=" * 50)

    trainer = ReconciliationTrainer(model_name=model_name)

    # Train the model
    trainer.train_model(
        dataset['training_examples'],
        output_dir="models/reconciliation_model",
        epochs=epochs
    )

    print("✓ Model training completed")
    return trainer

def evaluate_model(trainer, dataset):
    """Evaluate the trained model"""
    print("=" * 50)
    print("EVALUATING MODEL")
    print("=" * 50)

    # Use a subset for evaluation
    test_examples = dataset['training_examples'][:100]
    results = trainer.evaluate_model(test_examples)

    print(f"✓ Model accuracy: {results['accuracy']:.2%}")

    # Save evaluation results
    with open('results/evaluation_results.json', 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'num_test_examples': len(test_examples)
        }, f, indent=2)

    return results

def demo_model(trainer):
    """Demonstrate the model with sample data"""
    print("=" * 50)
    print("MODEL DEMONSTRATION")
    print("=" * 50)

    # Test cases
    test_cases = [
        {
            'name': 'Perfect Match',
            'tlf': {
                'settlement_date': '2024-01-15',
                'term_id': 'TERM1234',
                'orig_amt': 1500.00,
                'card_num': '412345678901',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000001',
                'trn_status': 'Completed'
            },
            'ilf_issuer': {
                'report_date': '2024-01-15',
                'term_id': 'TERM1234',
                'orig_amt': 1500.00,
                'card_num': '412345678901',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000001',
                'trn_status': 'Completed',
                'role': 'issuer'
            },
            'ilf_acquirer': {
                'report_date': '2024-01-15',
                'term_id': 'TERM1234',
                'orig_amt': 1500.00,
                'card_num': '412345678901',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000001',
                'trn_status': 'Completed',
                'role': 'acquirer'
            }
        },
        {
            'name': 'Amount Discrepancy',
            'tlf': {
                'settlement_date': '2024-01-16',
                'term_id': 'TERM1235',
                'orig_amt': 2000.00,
                'card_num': '412345678902',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000002',
                'trn_status': 'Completed'
            },
            'ilf_issuer': {
                'report_date': '2024-01-16',
                'term_id': 'TERM1235',
                'orig_amt': 2005.00,
                'card_num': '412345678902',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000002',
                'trn_status': 'Completed',
                'role': 'issuer'
            },
            'ilf_acquirer': {
                'report_date': '2024-01-16',
                'term_id': 'TERM1235',
                'orig_amt': 2005.00,
                'card_num': '412345678902',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000002',
                'trn_status': 'Completed',
                'role': 'acquirer'
            }
        },
        {
            'name': 'Missing in ILF',
            'tlf': {
                'settlement_date': '2024-01-17',
                'term_id': 'TERM1236',
                'orig_amt': 750.00,
                'card_num': '412345678903',
                'trn_desc': 'Purchase',
                'seq_num': 'SEQ_000003',
                'trn_status': 'Completed'
            },
            'ilf_issuer': None,
            'ilf_acquirer': None
        }
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        analysis = trainer.generate_reconciliation_report(
            test_case['tlf'],
            test_case['ilf_issuer'],
            test_case['ilf_acquirer']
        )
        print(f"Analysis: {analysis}")

def main():
    parser = argparse.ArgumentParser(description='Train Report Reconciliation SLM')
    parser.add_argument('--num-records', type=int, default=1000,
                       help='Number of training records to generate')
    parser.add_argument('--model-name', type=str, default="microsoft/DialoGPT-small",
                       help='Base model to use for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--skip-data-generation', action='store_true',
                       help='Skip data generation and use existing data')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing model')
    parser.add_argument('--demo-only', action='store_true',
                       help='Only run demonstration with existing model')

    args = parser.parse_args()

    # Setup directories
    setup_directories()

    # Load or generate data
    if args.demo_only or args.skip_data_generation:
        print("Loading existing training data...")
        try:
            with open('data/training_examples.json', 'r') as f:
                training_examples = json.load(f)

            dataset = {
                'training_examples': training_examples,
                'tlf': None,  # Not needed for demo
                'ilf': None   # Not needed for demo
            }
            print(f"✓ Loaded {len(training_examples)} training examples")
        except FileNotFoundError:
            print("❌ No existing training data found. Please run without --skip-data-generation first.")
            return
    else:
        dataset = generate_data(args.num_records)

    # Initialize trainer
    if args.demo_only:
        print("Loading existing model...")
        try:
            trainer = ReconciliationTrainer()
            trainer.model = trainer.model.__class__.from_pretrained("models/reconciliation_model")
            trainer.tokenizer = trainer.tokenizer.__class__.from_pretrained("models/reconciliation_model")
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Could not load existing model: {e}")
            print("Please train a model first by running without --demo-only")
            return
    else:
        # Train model
        if not args.skip_training:
            trainer = train_model(dataset, args.model_name, args.epochs)
        else:
            print("Loading existing model...")
            try:
                trainer = ReconciliationTrainer()
                trainer.model = trainer.model.__class__.from_pretrained("models/reconciliation_model")
                trainer.tokenizer = trainer.tokenizer.__class__.from_pretrained("models/reconciliation_model")
                print("✓ Model loaded successfully")
            except Exception as e:
                print(f"❌ Could not load existing model: {e}")
                return

        # Evaluate model
        if not args.demo_only:
            evaluate_model(trainer, dataset)

    # Run demonstration
    demo_model(trainer)

    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 50)
    print("✓ Model trained and saved to 'models/reconciliation_model'")
    print("✓ Training data saved to 'data/' directory")
    print("✓ Evaluation results saved to 'results/' directory")
    print("\nTo use the model:")
    print("1. Load the model from 'models/reconciliation_model'")
    print("2. Use ReconciliationTrainer.generate_reconciliation_report()")
    print("3. Check the demonstration examples above")

if __name__ == "__main__":
    main()
