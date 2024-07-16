import pandas as pd
import numpy as np
import random

df=pd.read_csv("label.csv")

# Define split proportions
healthy_split = 0.75
non_healthy_split = 0.7

# Initialize train and test DataFrames
train_df = pd.DataFrame(columns=['subject_id', 'diagnosis', 'split'])
test_df = pd.DataFrame(columns=['subject_id', 'diagnosis', 'split'])

# Loop through each diagnosis ('Control' and 'LS')
for diagnosis in ['Control', 'LS']:
    subset_df = df[df['diagnosis'] == diagnosis].copy()
    subset_df = subset_df.sample(frac=1).reset_index(drop=True)  # Shuffle subset
    
    # Calculate number of samples for train and test
    num_samples = len(subset_df)
    num_train = int(num_samples * healthy_split) if diagnosis == 'Control' else int(num_samples * non_healthy_split)
    num_test = num_samples - num_train
    
    # Assign subjects to train and test based on calculated splits
    train_subjects = subset_df.iloc[:num_train].copy()
    test_subjects = subset_df.iloc[num_train:num_train + num_test].copy()
    
    # Add 'split' column 
    train_subjects.loc[:, 'split'] = 'TRAIN'
    test_subjects.loc[:, 'split'] = 'TEST'
    
    # Append to train_df and test_df
    train_df = train_df.append(train_subjects, ignore_index=True)
    test_df = test_df.append(test_subjects, ignore_index=True)

# Ensure subject IDs are unique between train and test sets
train_ids = set(train_df['subject_id'])
test_ids = set(test_df['subject_id'])

# Check for common subject IDs and resolve if necessary
common_ids = train_ids.intersection(test_ids)
for subject_id in common_ids:
    if subject_id in train_ids:
        new_id = max(train_ids.union(test_ids)) + 1
        train_df.loc[train_df['subject_id'] == subject_id, 'subject_id'] = new_id
        train_ids.add(new_id)
    if subject_id in test_ids:
        new_id = max(train_ids.union(test_ids)) + 1
        test_df.loc[test_df['subject_id'] == subject_id, 'subject_id'] = new_id
        test_ids.add(new_id)

# Concatenate train and test DataFrames back together
df_final = pd.concat([train_df, test_df], ignore_index=True)

df_final.to_csv("data.csv")

# Check for uniqueness of subject IDs across train and test sets
train_ids_set = set(train_df['subject_id'])
test_ids_set = set(test_df['subject_id'])

if train_ids_set.isdisjoint(test_ids_set):
    print("Subject IDs are unique between train and test sets.")
else:
    print("Warning: Subject IDs are not unique between train and test sets!")

# Count number of samples with each diagnosis in train and test sets
train_counts = train_df['diagnosis'].value_counts()
test_counts = test_df['diagnosis'].value_counts()

print("Number of samples with each diagnosis in TRAIN set:")
print(train_counts)
print()

print("Number of samples with each diagnosis in TEST set:")
print(test_counts)

train_ids_count = train_df['subject_id'].nunique()
test_ids_count = test_df['subject_id'].nunique()

print("Number of unique subject IDs in TRAIN set:", train_ids_count)
print("Number of unique subject IDs in TEST set:", test_ids_count)