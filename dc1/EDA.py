#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


# Change to parent directory
path = Path.cwd()
os.chdir(path.parent)


# In[ ]:


# Import the data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/Y_train.npy')
y_test = np.load('data/Y_test.npy')


# ## Format checks

# In[ ]:


# Do datasets have the same length?
def check_shapes(train, test):
    print(f'equal cardinality: {train.shape[0] == test.shape[0]}')

check_shapes(X_train, y_train) # Train data
check_shapes(X_test, y_test)   # Test data


# In[ ]:


# Data overview: counts, unique values statistics
unique_vals, counts = np.unique(y_train, return_counts=True)

print(f'Total train: \n {len(y_train)}')
print(f'Categories: \n {unique_vals}')
print(f'Counts: \n {counts}')


# In[ ]:


sns.barplot(counts)


# In[ ]:


# Check if all images have the same format
image_expected_shape = (128, 128)

def expected_shape(image_dataset, expected_size) -> None:
    same_shape = True
    for datapoint in image_dataset:
        if datapoint[0].shape != expected_size:
            same_shape = False
    print(f'expected shape: {same_shape}')

expected_shape(X_train, image_expected_shape) # Training set
expected_shape(X_test, image_expected_shape)  # Testing set


# ### Is the data Normalized?

# In[ ]:


# Calculate MEAN and STD
max_values = list()
min_values = list()
for datapoint in X_train:
    max_values.append(np.max(datapoint[0]))
    min_values.append(np.min(datapoint[0]))


# "The most common pixel format is the byte image, where this number is stored as an 8-bit integer giving a range of possible values from 0 to 255. Typically zero is taken to be black, and 255 is taken to be white. Values in between make up the different shades of gray. Thus only integers are stored."

# In[ ]:


np.max(max_values), np.min(min_values)


# ### Are there any duplicates in the datasets?

# In[ ]:


def duplicates_(dataset):
    num_duplicates = 0
    seen = list()
    for datapoint in dataset:
        seen.append(datapoint[0])

    j = 1
    for i, _ in enumerate(seen):
        while j < len(seen):
            if (i != j) and (np.array_equal(seen[i], seen[j], equal_nan=True)) :
                num_duplicates += 1
            j += 1
    print(f'Number of duplicates: {num_duplicates}')

duplicates_(X_train)
duplicates_(X_test)       


# ## Distribution of data

# In[ ]:


# Calculate MEAN and STD
mean_values = list()
std_values = list()
for datapoint in X_train:
    mean_values.append(np.mean(datapoint[0]))
    std_values.append(np.std(datapoint[0]))


# In[ ]:


# Plot the distribution of general image darkness
plt.hist(mean_values);


# In[ ]:


# Distribution of image contrasts.
plt.hist(std_values);


# ## Check data distributions per category

# In[ ]:


# Check distributions per category
def dist_per_cat(mean_data, std_data, y_data):
    label_index_dict = {0: list(), 
                            1: list(), 
                            2: list(), 
                            3: list(), 
                            4: list(), 
                            5: list()}
    mean_per_cat = label_index_dict.copy()
    std_per_cat = label_index_dict.copy()

    for j in range(6):
        for label in y_data:
            if label == j:
                label_index_dict[j].append(True)
            else:
                label_index_dict[j].append(False)
            
    for cat in range(6): 
        mean_per_cat[cat] = [val for in_cat, val in zip(label_index_dict[cat], mean_data) if in_cat]

    for cat in range(6): 
        std_per_cat[cat] = [val for in_cat, val in zip(label_index_dict[cat], std_data) if in_cat]

    return mean_per_cat, std_per_cat

mean_per_cat, std_per_cat = dist_per_cat(mean_values, std_values, y_train)


# In[ ]:


mean_per_cat.keys(), 
for val in mean_per_cat.values():
    print(len(val))


# In[ ]:


sns.boxplot(data=mean_per_cat);


# In[ ]:


sns.boxplot(data=std_per_cat);


# ### End of notebook
# // Alicia Larsen
# // Conversion to .py by Giuseppe Vescina
