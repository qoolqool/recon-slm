import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import wandb
import re

class ReconciliationTrainer:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add special tokens for reconciliation
        special_tokens = {
            "additional_special_tokens": [
                "[TLF]", "[ILF_ISSUER]", "[ILF_ACQUIRER]", "[MATCH]", "[MISMATCH]",
                "[AMOUNT_DISCREPANCY]", "[DATE_DISCREPANCY]", "[STATUS_DISCREPANCY]",
                "[MISSING_IN_ILF]", "[UNMATCHED_ILF]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self, training_examples):
        """Prepare dataset for training"""
        # Format the data for causal language modeling
        formatted_data = []

        for example in training_examples:
            # Create a formatted training example
            formatted_text = f"<|startoftext|>Reconciliation Task:\n{example['input']}\n\nAnalysis: {example['output']}<|endoftext|>"
            formatted_data.append(formatted_text)

        # Tokenize the data
        tokenized_data = self.tokenizer(
            formatted_data,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'labels': tokenized_data['input_ids'].clone()
        })

        return dataset

    def train_model(self, training_examples, output_dir="./reconciliation_model", epochs=3):
        """Train the model"""

        # Prepare dataset
        full_dataset = self.prepare_dataset(training_examples)

        # Split into train/validation
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        print(f"Total dataset size: {dataset_size}")
        print(f"Train size: {train_size}, Validation size: {val_size}")

        train_dataset = full_dataset.select(range(train_size))
        val_dataset = full_dataset.select(range(train_size, dataset_size)) if val_size > 0 else None

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy="steps" if val_dataset is not None else "no",
            eval_steps=200 if val_dataset is not None else None,
            save_steps=400,
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset is not None else False,
            metric_for_best_model="eval_loss" if val_dataset is not None else None,
            greater_is_better=False,
            dataloader_num_workers=0,
            fp16=False,
            gradient_checkpointing=True,
            learning_rate=5e-5,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")

        return trainer

    def generate_reconciliation_report(self, tlf_record, ilf_issuer_record=None, ilf_acquirer_record=None):
        """Generate reconciliation analysis for given records"""

        # Prepare input text
        if ilf_issuer_record is not None and ilf_acquirer_record is not None:
            input_text = f"""<|startoftext|>Reconciliation Task:
Reconcile the following transactions:
[TLF] Date={tlf_record.get('settlement_date', 'N/A')}, TermID={tlf_record.get('term_id', 'N/A')}, Amount={tlf_record.get('orig_amt', 'N/A')}, Card={tlf_record.get('card_num', 'N/A')}, Status={tlf_record.get('trn_status', 'N/A')}
[ILF_ISSUER] Date={ilf_issuer_record.get('report_date', 'N/A')}, TermID={ilf_issuer_record.get('term_id', 'N/A')}, Amount={ilf_issuer_record.get('orig_amt', 'N/A')}, Card={ilf_issuer_record.get('card_num', 'N/A')}, Status={ilf_issuer_record.get('trn_status', 'N/A')}
[ILF_ACQUIRER] Date={ilf_acquirer_record.get('report_date', 'N/A')}, TermID={ilf_acquirer_record.get('term_id', 'N/A')}, Amount={ilf_acquirer_record.get('orig_amt', 'N/A')}, Card={ilf_acquirer_record.get('card_num', 'N/A')}, Status={ilf_acquirer_record.get('trn_status', 'N/A')}

Analysis: """
        else:
            input_text = f"""<|startoftext|>Reconciliation Task:
Reconcile the following transactions:
[TLF] Date={tlf_record.get('settlement_date', 'N/A')}, TermID={tlf_record.get('term_id', 'N/A')}, Amount={tlf_record.get('orig_amt', 'N/A')}, Card={tlf_record.get('card_num', 'N/A')}, Status={tlf_record.get('trn_status', 'N/A')}
[ILF_ISSUER] No matching record found
[ILF_ACQUIRER] No matching record found

Analysis: """

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=400)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the analysis part
        analysis_start = response.find("Analysis: ") + len("Analysis: ")
        analysis = response[analysis_start:].strip()

        return analysis

    def evaluate_model(self, test_examples):
        """Evaluate the model on test examples"""
        correct_predictions = 0
        total_predictions = len(test_examples)

        predictions = []
        actual = []

        def parse_record(line):
            """Parse a record line into a dictionary"""
            if 'No matching record found' in line:
                return None
            # Example line: [TLF] Date=2024-01-01, TermID=TERM1234, Amount=1000.0, Card=412345678901, Status=Completed
            pattern = r".*Date=([^,]+), TermID=([^,]+), Amount=([^,]+), Card=([^,]+), Status=([^,]+)"
            match = re.match(pattern, line)
            if match:
                return {
                    'date': match.group(1),
                    'term_id': match.group(2),
                    'amount': float(match.group(3)) if match.group(3) != 'N/A' else 'N/A',
                    'card_num': match.group(4),
                    'status': match.group(5)
                }
            return None

        for example in test_examples:
            # Parse the input to extract system records
            input_lines = example['input'].split('\n')
            tlf_line = next((line for line in input_lines if '[TLF]' in line), None)
            ilf_issuer_line = next((line for line in input_lines if '[ILF_ISSUER]' in line), None)
            ilf_acquirer_line = next((line for line in input_lines if '[ILF_ACQUIRER]' in line), None)

            if not tlf_line:
                print(f"Warning: No TLF line found in input: {example['input']}")
                continue

            # Parse records
            tlf_record = parse_record(tlf_line)
            ilf_issuer_record = parse_record(ilf_issuer_line) if ilf_issuer_line else None
            ilf_acquirer_record = parse_record(ilf_acquirer_line) if ilf_acquirer_line else None

            # Rename 'date' to match expected keys in generate_reconciliation_report
            if tlf_record:
                tlf_record['settlement_date'] = tlf_record.pop('date')
                tlf_record['orig_amt'] = tlf_record.pop('amount')
                tlf_record['trn_status'] = tlf_record.pop('status')
            if ilf_issuer_record:
                ilf_issuer_record['report_date'] = ilf_issuer_record.pop('date')
                ilf_issuer_record['orig_amt'] = ilf_issuer_record.pop('amount')
                ilf_issuer_record['trn_status'] = ilf_issuer_record.pop('status')
            if ilf_acquirer_record:
                ilf_acquirer_record['report_date'] = ilf_acquirer_record.pop('date')
                ilf_acquirer_record['orig_amt'] = ilf_acquirer_record.pop('amount')
                ilf_acquirer_record['trn_status'] = ilf_acquirer_record.pop('status')

            # Generate prediction
            prediction = self.generate_reconciliation_report(tlf_record, ilf_issuer_record, ilf_acquirer_record)

            predictions.append(prediction)
            actual.append(example['output'])

            # Check accuracy based on mismatch type
            if example['category'].replace('_', ' ').lower() in prediction.lower() or ('match' in prediction.lower() and example['category'] == 'match'):
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Model Accuracy: {accuracy:.2%}")

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'actual': actual
        }

# Usage example
def main():
    # Initialize trainer
    trainer = ReconciliationTrainer()

    # Load training data
    with open('data/training_examples.json', 'r') as f:
        training_examples = json.load(f)

    print(f"Loaded {len(training_examples)} training examples")

    # Train the model
    trainer.train_model(training_examples, epochs=3)

    # Test the model
    test_examples = training_examples[:50]  # Use first 50 for testing
    results = trainer.evaluate_model(test_examples)

    print(f"Training completed with {results['accuracy']:.2%} accuracy on test set")

    # Example usage
    sample_tlf = {
        'settlement_date': '2024-01-15',
        'term_id': 'TERM1234',
        'orig_amt': 1500.00,
        'card_num': '412345678901',
        'trn_desc': 'Purchase',
        'seq_num': 'SEQ_000001',
        'trn_status': 'Completed'
    }

    sample_ilf_issuer = {
        'report_date': '2024-01-15',
        'term_id': 'TERM1234',
        'orig_amt': 1505.00,
        'card_num': '412345678901',
        'trn_desc': 'Purchase',
        'seq_num': 'SEQ_000001',
        'trn_status': 'Completed',
        'role': 'issuer'
    }

    sample_ilf_acquirer = {
        'report_date': '2024-01-15',
        'term_id': 'TERM1234',
        'orig_amt': 1505.00,
        'card_num': '412345678901',
        'trn_desc': 'Purchase',
        'seq_num': 'SEQ_000001',
        'trn_status': 'Completed',
        'role': 'acquirer'
    }

    analysis = trainer.generate_reconciliation_report(sample_tlf, sample_ilf_issuer, sample_ilf_acquirer)
    print(f"\nSample Analysis: {analysis}")

if __name__ == "__main__":
    main()
