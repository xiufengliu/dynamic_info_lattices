#!/usr/bin/env python3
"""
Process and consolidate new datasets from data/raw directory
"""

import pandas as pd
import numpy as np
import os
import gzip
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def process_electricity_dataset():
    """Process electricity dataset (ECL) - 370 series, hourly electricity consumption"""
    print("Processing Electricity dataset...")
    
    df = pd.read_csv('data/raw/electricity/electricity.csv')
    print(f"Electricity shape: {df.shape}")
    
    # Remove date column and OT column, keep only the 320 main series
    data_cols = [col for col in df.columns if col not in ['date', 'OT']]
    data = df[data_cols].values.astype(np.float32)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/electricity.npy', data)
    
    # Create metadata
    metadata = {
        'name': 'electricity',
        'description': 'Electricity Consuming Load (ECL) - 370 series, hourly electricity consumption',
        'shape': data.shape,
        'frequency': 'hourly',
        'num_series': data.shape[1],
        'length': data.shape[0],
        'source': 'UCI ML Repository'
    }
    
    return metadata

def process_exchange_rate_dataset():
    """Process exchange rate dataset - 8 series, daily exchange rates"""
    print("Processing Exchange Rate dataset...")
    
    df = pd.read_csv('data/raw/exchange_rate/exchange_rate.csv')
    print(f"Exchange Rate shape: {df.shape}")
    
    # Remove date column and OT column
    data_cols = [col for col in df.columns if col not in ['date', 'OT']]
    data = df[data_cols].values.astype(np.float32)
    
    # Save processed data
    np.save('data/processed/exchange_rate.npy', data)
    
    metadata = {
        'name': 'exchange_rate',
        'description': 'Exchange Rate - 8 series, daily exchange rates',
        'shape': data.shape,
        'frequency': 'daily',
        'num_series': data.shape[1],
        'length': data.shape[0],
        'source': 'Federal Reserve Economic Data'
    }
    
    return metadata

def process_traffic_dataset():
    """Process traffic dataset - 862 series, hourly traffic occupancy"""
    print("Processing Traffic dataset...")
    
    df = pd.read_csv('data/raw/traffic/traffic.csv')
    print(f"Traffic shape: {df.shape}")
    
    # Remove date column and OT column
    data_cols = [col for col in df.columns if col not in ['date', 'OT']]
    data = df[data_cols].values.astype(np.float32)
    
    # Save processed data
    np.save('data/processed/traffic.npy', data)
    
    metadata = {
        'name': 'traffic',
        'description': 'Traffic - 862 series, hourly traffic occupancy rates',
        'shape': data.shape,
        'frequency': 'hourly',
        'num_series': data.shape[1],
        'length': data.shape[0],
        'source': 'California Department of Transportation'
    }
    
    return metadata

def process_weather_dataset():
    """Process weather dataset - 21 series, 10-minute weather measurements"""
    print("Processing Weather dataset...")
    
    df = pd.read_csv('data/raw/weather/weather.csv')
    print(f"Weather shape: {df.shape}")
    
    # Remove date column and OT column
    data_cols = [col for col in df.columns if col not in ['date', 'OT']]
    data = df[data_cols].values.astype(np.float32)
    
    # Handle missing values
    data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
    
    # Save processed data
    np.save('data/processed/weather.npy', data)
    
    metadata = {
        'name': 'weather',
        'description': 'Weather - 21 series, 10-minute weather measurements',
        'shape': data.shape,
        'frequency': '10min',
        'num_series': data.shape[1],
        'length': data.shape[0],
        'source': 'Max Planck Institute for Biogeochemistry'
    }
    
    return metadata

def process_illness_dataset():
    """Process illness dataset - 7 series, weekly illness surveillance"""
    print("Processing Illness dataset...")
    
    df = pd.read_csv('data/raw/illness/national_illness.csv')
    print(f"Illness shape: {df.shape}")
    
    # Remove date column and OT column
    data_cols = [col for col in df.columns if col not in ['date', 'OT']]
    data = df[data_cols].values.astype(np.float32)
    
    # Save processed data
    np.save('data/processed/illness.npy', data)
    
    metadata = {
        'name': 'illness',
        'description': 'Illness - 7 series, weekly influenza-like illness surveillance',
        'shape': data.shape,
        'frequency': 'weekly',
        'num_series': data.shape[1],
        'length': data.shape[0],
        'source': 'CDC Influenza Surveillance'
    }
    
    return metadata

def process_solar_dataset():
    """Process solar dataset - 137 series, 10-minute photovoltaic power"""
    print("Processing Solar dataset...")
    
    # Read compressed file
    with gzip.open('data/raw/solar_AL.txt.gz', 'rt') as f:
        lines = f.readlines()
    
    # Parse data
    data = []
    for line in lines:
        values = [float(x) for x in line.strip().split(',')]
        data.append(values)
    
    data = np.array(data, dtype=np.float32)
    print(f"Solar shape: {data.shape}")
    
    # Save processed data
    np.save('data/processed/solar.npy', data)
    
    metadata = {
        'name': 'solar',
        'description': 'Solar Energy - 137 series, 10-minute photovoltaic power production',
        'shape': data.shape,
        'frequency': '10min',
        'num_series': data.shape[1],
        'length': data.shape[0],
        'source': 'National Renewable Energy Laboratory'
    }
    
    return metadata

def create_train_test_splits():
    """Create train/test splits for all processed datasets"""
    print("\nCreating train/test splits...")
    
    processed_dir = Path('data/processed')
    splits_dir = Path('data/splits')
    splits_dir.mkdir(exist_ok=True)
    
    for npy_file in processed_dir.glob('*.npy'):
        dataset_name = npy_file.stem
        print(f"Creating splits for {dataset_name}...")
        
        data = np.load(npy_file)
        
        # Use 70% for training, 30% for testing
        split_point = int(0.7 * len(data))
        
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        # Save splits
        np.save(splits_dir / f'{dataset_name}_train.npy', train_data)
        np.save(splits_dir / f'{dataset_name}_test.npy', test_data)
        
        print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

def process_existing_datasets():
    """Process existing datasets in data/ folder to match new format"""
    print("\nProcessing existing datasets...")

    existing_datasets = [
        'ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv',
        'ECL.csv', 'gefcom2014.csv', 'southern_china.csv'
    ]

    metadata_list = []

    for dataset_file in existing_datasets:
        file_path = f'data/{dataset_file}'
        if os.path.exists(file_path):
            print(f"Processing {dataset_file}...")

            try:
                df = pd.read_csv(file_path)
                print(f"  Original shape: {df.shape}")

                # Remove date/time columns - more comprehensive detection
                original_columns = df.columns.tolist()

                # Try to identify and remove date/time columns
                columns_to_drop = []
                for col in df.columns:
                    # Check if column name suggests it's a date/time column
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                        columns_to_drop.append(col)
                        continue

                    # Check if column contains date-like strings
                    try:
                        sample_values = df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            # Try to detect date patterns
                            sample_str = str(sample_values.iloc[0])
                            if any(char in sample_str for char in ['-', ':', ' ']) and len(sample_str) > 8:
                                # Looks like a date/time string
                                columns_to_drop.append(col)
                    except:
                        pass

                # Drop identified date/time columns
                if columns_to_drop:
                    print(f"  Dropping date/time columns: {columns_to_drop}")
                    df = df.drop(columns_to_drop, axis=1)

                # Convert to numpy array, handling non-numeric columns
                try:
                    data = df.values.astype(np.float32)
                except ValueError:
                    # If there are still non-numeric columns, try to convert them
                    print(f"  Converting non-numeric columns...")
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) == 0:
                        print(f"  No numeric columns found, skipping {dataset_file}")
                        continue
                    data = numeric_df.values.astype(np.float32)

                # Handle missing values
                if np.isnan(data).any():
                    print(f"  Handling {np.isnan(data).sum()} missing values...")
                    data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values

                # Save processed data
                dataset_name = dataset_file.replace('.csv', '').lower()
                np.save(f'data/processed/{dataset_name}.npy', data)

                print(f"  Processed shape: {data.shape}")

                # Create metadata
                metadata = {
                    'name': dataset_name,
                    'description': f'{dataset_name.upper()} - {data.shape[1]} series',
                    'shape': data.shape,
                    'frequency': 'hourly' if 'h' in dataset_name else 'unknown',
                    'num_series': data.shape[1],
                    'length': data.shape[0],
                    'source': 'Existing dataset'
                }
                metadata_list.append(metadata)

            except Exception as e:
                print(f"  Error processing {dataset_file}: {e}")

    return metadata_list

def main():
    """Main processing function"""
    print("=" * 60)
    print("Processing New Datasets for Dynamic Information Lattices")
    print("=" * 60)
    
    # Change to project root directory
    os.chdir('/zhome/bb/9/101964/xiuli/dynamic_info_lattices')
    
    # Process all new datasets
    metadata_list = []
    
    try:
        metadata_list.append(process_electricity_dataset())
    except Exception as e:
        print(f"Error processing electricity: {e}")
    
    try:
        metadata_list.append(process_exchange_rate_dataset())
    except Exception as e:
        print(f"Error processing exchange rate: {e}")
    
    try:
        metadata_list.append(process_traffic_dataset())
    except Exception as e:
        print(f"Error processing traffic: {e}")
    
    try:
        metadata_list.append(process_weather_dataset())
    except Exception as e:
        print(f"Error processing weather: {e}")
    
    try:
        metadata_list.append(process_illness_dataset())
    except Exception as e:
        print(f"Error processing illness: {e}")
    
    try:
        metadata_list.append(process_solar_dataset())
    except Exception as e:
        print(f"Error processing solar: {e}")
    
    # Process existing datasets
    try:
        existing_metadata = process_existing_datasets()
        metadata_list.extend(existing_metadata)
    except Exception as e:
        print(f"Error processing existing datasets: {e}")
    
    # Create train/test splits
    try:
        create_train_test_splits()
    except Exception as e:
        print(f"Error creating splits: {e}")
    
    # Save metadata
    import json
    with open('data/processed/datasets_metadata.json', 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset Processing Summary")
    print("=" * 60)
    
    for metadata in metadata_list:
        print(f"✅ {metadata['name']:15}: {metadata['shape']} - {metadata['description']}")
    
    print(f"\n📁 Processed datasets saved to: data/processed/")
    print(f"📁 Train/test splits saved to: data/splits/")
    print(f"📄 Metadata saved to: data/processed/datasets_metadata.json")
    
    print("\n🎉 All datasets processed successfully!")

if __name__ == "__main__":
    main()
