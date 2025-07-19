import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class ReconciliationDataGenerator:
    def __init__(self):
        self.companies = ['ABC Corp', 'XYZ Ltd', 'Tech Solutions', 'Global Trade', 'Finance Plus']
        self.transaction_descriptions = ['Purchase', 'Refund', 'Cash Advance', 'Payment', 'Fee']
        self.card_numbers = [f'4{random.randint(100000000000, 999999999999)}' for _ in range(100)]
        self.terminal_ids = [f'TERM{random.randint(1000, 9999)}' for _ in range(50)]
        self.fiids = ['FIID1', 'FIID2', 'FIID3', 'FIID4']

    def generate_tlf_data(self, num_records=1000):
        """Generate data for TLF system (1 row per transaction)"""
        data = []
        base_date = datetime(2024, 1, 1)

        for i in range(num_records):
            settlement_date = base_date + timedelta(days=random.randint(0, 365))
            record = {
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'card_fiid': random.choice(self.fiids),
                'term_fiid': random.choice(self.fiids),
                'term_id': random.choice(self.terminal_ids),
                'orig_amt': round(random.uniform(100, 50000), 2),
                'card_num': random.choice(self.card_numbers),
                'trn_desc': random.choice(self.transaction_descriptions),
                'seq_num': f'SEQ_{i+1:06d}',
                'trn_status': random.choice(['Completed', 'Pending', 'Failed'])
            }
            data.append(record)

        return pd.DataFrame(data)

    def generate_ilf_data(self, tlf_data, mismatch_rate=0.15):
        """Generate data for ILF system (2 rows per transaction: issuer and acquirer)"""
        data = []
        base_date = datetime(2024, 1, 1)  # Define base_date for ILF extra records

        for _, row in tlf_data.iterrows():
            # Create base records (issuer and acquirer)
            settlement_date = datetime.strptime(row['settlement_date'], '%Y-%m-%d')
            
            issuer_record = {
                'report_date': row['settlement_date'],
                'participant': row['card_fiid'],
                'fi_nm': random.choice(self.companies),
                'term_id': row['term_id'],
                'orig_amt': row['orig_amt'],
                'card_num': row['card_num'],
                'trn_desc': row['trn_desc'],
                'seq_num': row['seq_num'],
                'trn_status': row['trn_status'],
                'role': 'issuer',
                'mismatch_type': 'match',
                'mismatch_reason': 'Perfect match'
            }

            acquirer_record = {
                'report_date': row['settlement_date'],
                'participant': row['term_fiid'],
                'fi_nm': random.choice(self.companies),
                'term_id': row['term_id'],
                'orig_amt': row['orig_amt'],
                'card_num': row['card_num'],
                'trn_desc': row['trn_desc'],
                'seq_num': row['seq_num'],
                'trn_status': row['trn_status'],
                'role': 'acquirer',
                'mismatch_type': 'match',
                'mismatch_reason': 'Perfect match'
            }

            # Introduce mismatches randomly
            if random.random() < mismatch_rate:
                mismatch_type = random.choice(['amount', 'date', 'status', 'missing'])

                if mismatch_type == 'amount':
                    variance = random.uniform(0.95, 1.05)
                    issuer_record['orig_amt'] = round(issuer_record['orig_amt'] * variance, 2)
                    acquirer_record['orig_amt'] = issuer_record['orig_amt']
                    issuer_record['mismatch_type'] = 'amount_discrepancy'
                    issuer_record['mismatch_reason'] = 'Possible rounding difference or fee adjustment'
                    acquirer_record['mismatch_type'] = 'amount_discrepancy'
                    acquirer_record['mismatch_reason'] = 'Possible rounding difference or fee adjustment'

                elif mismatch_type == 'date':
                    new_date = settlement_date + timedelta(days=random.randint(1, 3))
                    issuer_record['report_date'] = new_date.strftime('%Y-%m-%d')
                    acquirer_record['report_date'] = new_date.strftime('%Y-%m-%d')
                    issuer_record['mismatch_type'] = 'date_discrepancy'
                    issuer_record['mismatch_reason'] = 'Processing delay or different recording date'
                    acquirer_record['mismatch_type'] = 'date_discrepancy'
                    acquirer_record['mismatch_reason'] = 'Processing delay or different recording date'

                elif mismatch_type == 'status':
                    new_status = random.choice(['Completed', 'Pending', 'Failed'])
                    issuer_record['trn_status'] = new_status
                    acquirer_record['trn_status'] = new_status
                    issuer_record['mismatch_type'] = 'status_discrepancy'
                    issuer_record['mismatch_reason'] = 'Different processing status'
                    acquirer_record['mismatch_type'] = 'status_discrepancy'
                    acquirer_record['mismatch_reason'] = 'Different processing status'

                elif mismatch_type == 'missing':
                    continue

            data.append(issuer_record)
            data.append(acquirer_record)

        # Add some extra records in ILF (unmatched)
        for i in range(int(len(tlf_data) * 0.05)):
            extra_date = base_date + timedelta(days=random.randint(0, 365))
            extra_record_issuer = {
                'report_date': extra_date.strftime('%Y-%m-%d'),
                'participant': random.choice(self.fiids),
                'fi_nm': random.choice(self.companies),
                'term_id': random.choice(self.terminal_ids),
                'orig_amt': round(random.uniform(100, 50000), 2),
                'card_num': random.choice(self.card_numbers),
                'trn_desc': random.choice(self.transaction_descriptions),
                'seq_num': f'SEQ_EXTRA_{i+1:03d}',
                'trn_status': random.choice(['Completed', 'Pending']),
                'role': 'issuer',
                'mismatch_type': 'unmatched_ilf',
                'mismatch_reason': 'Transaction exists only in ILF system'
            }
            extra_record_acquirer = extra_record_issuer.copy()
            extra_record_acquirer['participant'] = random.choice(self.fiids)
            extra_record_acquirer['role'] = 'acquirer'
            extra_record_acquirer['mismatch_type'] = 'unmatched_ilf'
            extra_record_acquirer['mismatch_reason'] = 'Transaction exists only in ILF system'

            data.append(extra_record_issuer)
            data.append(extra_record_acquirer)

        return pd.DataFrame(data)

    def generate_training_examples(self, tlf_data, ilf_data):
        """Generate training examples for reconciliation"""
        training_data = []

        for _, row_tlf in tlf_data.iterrows():
            # Create composite key for matching
            tlf_key = (
                row_tlf['settlement_date'],
                row_tlf['term_id'],
                row_tlf['orig_amt'],
                row_tlf['card_num'],
                row_tlf['trn_desc'],
                row_tlf['seq_num']
            )

            # Find matching records in ILF
            matching_ilf = ilf_data[
                (ilf_data['report_date'] == row_tlf['settlement_date']) &
                (ilf_data['term_id'] == row_tlf['term_id']) &
                (ilf_data['orig_amt'] == row_tlf['orig_amt']) &
                (ilf_data['card_num'] == row_tlf['card_num']) &
                (ilf_data['trn_desc'] == row_tlf['trn_desc']) &
                (ilf_data['seq_num'] == row_tlf['seq_num'])
            ]

            if not matching_ilf.empty:
                issuer_row = matching_ilf[matching_ilf['role'] == 'issuer'].iloc[0]
                acquirer_row = matching_ilf[matching_ilf['role'] == 'acquirer'].iloc[0]

                input_text = f"""
Reconcile the following transactions:
[TLF] Date={row_tlf['settlement_date']}, TermID={row_tlf['term_id']}, Amount={row_tlf['orig_amt']}, Card={row_tlf['card_num']}, Status={row_tlf['trn_status']}
[ILF_ISSUER] Date={issuer_row['report_date']}, TermID={issuer_row['term_id']}, Amount={issuer_row['orig_amt']}, Card={issuer_row['card_num']}, Status={issuer_row['trn_status']}
[ILF_ACQUIRER] Date={acquirer_row['report_date']}, TermID={acquirer_row['term_id']}, Amount={acquirer_row['orig_amt']}, Card={acquirer_row['card_num']}, Status={acquirer_row['trn_status']}
"""

                if issuer_row['mismatch_type'] == 'match':
                    output_text = "MATCH: Transactions are identical across both systems."
                else:
                    output_text = f"MISMATCH: {issuer_row['mismatch_type']} - {issuer_row['mismatch_reason']}"

                training_data.append({
                    'input': input_text.strip(),
                    'output': output_text,
                    'category': issuer_row['mismatch_type']
                })
            else:
                input_text = f"""
Reconcile the following transactions:
[TLF] Date={row_tlf['settlement_date']}, TermID={row_tlf['term_id']}, Amount={row_tlf['orig_amt']}, Card={row_tlf['card_num']}, Status={row_tlf['trn_status']}
[ILF_ISSUER] No matching record found
[ILF_ACQUIRER] No matching record found
"""
                output_text = "MISMATCH: missing_in_ilf - Transaction exists in TLF but not in ILF"

                training_data.append({
                    'input': input_text.strip(),
                    'output': output_text,
                    'category': 'missing_in_ilf'
                })

        return training_data

    def generate_full_dataset(self, num_records=1000):
        """Generate complete dataset"""
        print("Generating TLF data...")
        tlf_data = self.generate_tlf_data(num_records)

        print("Generating ILF data...")
        ilf_data = self.generate_ilf_data(tlf_data)

        print("Generating training examples...")
        training_examples = self.generate_training_examples(tlf_data, ilf_data)

        return {
            'tlf': tlf_data,
            'ilf': ilf_data,
            'training_examples': training_examples
        }

if __name__ == "__main__":
    generator = ReconciliationDataGenerator()

    # Generate dataset
    dataset = generator.generate_full_dataset(1000)

    # Save to files
    dataset['tlf'].to_csv('tlf_data.csv', index=False)
    dataset['ilf'].to_csv('ilf_data.csv', index=False)

    # Save training examples as JSON
    with open('training_examples.json', 'w') as f:
        json.dump(dataset['training_examples'], f, indent=2)

    print(f"Generated {len(dataset['training_examples'])} training examples")
    print(f"TLF records: {len(dataset['tlf'])}")
    print(f"ILF records: {len(dataset['ilf'])}")

    # Show example
    print("\nExample training pair:")
    example = dataset['training_examples'][0]
    print("Input:", example['input'])
    print("Output:", example['output'])
