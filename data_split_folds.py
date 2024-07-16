import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold

df=pd.read_csv("data.csv")

# Filter the DataFrame for 'TRAIN' split
train_df = df[df['split'] == 'TRAIN']

# Initialize the GroupKFold object
gkf = GroupKFold(n_splits=4)

# Prepare for cross-validation
X = train_df[['image', 'subject_id']]
y = train_df['diagnosis']
groups = train_df['subject_id']

# Create an empty DataFrame to store all folds
all_folds_df = pd.DataFrame()

# Perform the group-wise splitting
for fold_index, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
    fold_train_df = train_df.iloc[train_index].copy()
    fold_test_df = train_df.iloc[test_index].copy()

    # Adding columns to indicate the fold assignment
    fold_train_df['fold'] = fold_index
    fold_test_df['fold'] = fold_index

    # Concatenate to the overall folds DataFrame
    fold_concat_df = pd.concat([fold_train_df, fold_test_df], ignore_index=True)
    all_folds_df = pd.concat([all_folds_df, fold_concat_df], ignore_index=True)

# Create directories to store the data
base_output_dir = './data_split'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Iterate over each fold in the DataFrame
for fold_index in all_folds_df['fold'].unique():
    fold_dir = os.path.join(base_output_dir, f'fold_{fold_index}')
    os.makedirs(fold_dir, exist_ok=True)

    # Create train and valid directories
    train_dir = os.path.join(fold_dir, 'train')
    valid_dir = os.path.join(fold_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Create subdirectories for each unique diagnosis in train and valid
    for diagnosis_category in all_folds_df['diagnosis'].unique():
        train_diagnosis_dir = os.path.join(train_dir, diagnosis_category)
        valid_diagnosis_dir = os.path.join(valid_dir, diagnosis_category)
        os.makedirs(train_diagnosis_dir, exist_ok=True)
        os.makedirs(valid_diagnosis_dir, exist_ok=True)

    # Copy images to respective directories based on fold and diagnosis
    for _, row in all_folds_df.iterrows():
        src_path = row['image']
        if row['fold'] == fold_index:
            dst_dir = os.path.join(valid_dir, row['diagnosis'])
        else:
            dst_dir = os.path.join(train_dir, row['diagnosis'])
        dst_path = os.path.join(dst_dir, os.path.basename(row['image']))
        shutil.copy(src_path, dst_path)

print("Data split and copied successfully!")