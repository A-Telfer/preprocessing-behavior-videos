#!/usr/bin/env python
# coding: utf-8

# # Cleaning up Datasheets Example
# 
# Along with the videos, other forms of experimental data often need to be preprocessed
# 
# ## What's covered here
# This notebook demonstrates working with table data for treatment group and manual scores from a small experiment.
# - The data was split into multiple files
# - The data is not consistently formatted
# - There is some missing data that we were given in an email that we need to update manually
# 
# For an experiment this size, it would be pretty easy to input things manually. However, using this method we can easily validate there are no errors (can be easy to miss such as an extra space) and it's a useful demonstration for larger datasets.
# 

# ## Loading the Data

# In[1]:


import pandas as pd
from pathlib import Path

EXPERIMENT_DIR = Path("/home/andretelfer/shared/curated/fran/raw/")

# I found the sheet names by looking through the excel files themselves
cohort1 = pd.read_excel(EXPERIMENT_DIR / 'BehaviourData_Cohorts1&2.xlsx', sheet_name='Cohort 1')
cohort2 = pd.read_excel(EXPERIMENT_DIR / 'BehaviourData_Cohorts1&2.xlsx', sheet_name='Cohort 2')
cohort3a = pd.read_excel(EXPERIMENT_DIR / 'cohort 3.xlsx')
cohort3b = pd.read_excel(EXPERIMENT_DIR / 'Cohort 3 - Part 2.xlsx')


# ## Correcting Inconsistencies

# ### Checking Column Names
# 

# In[2]:


for df in [cohort1, cohort2, cohort3a, cohort3b]:
    print(df.columns.tolist())


# There are inconsistencies between column names, but they look like they'll be pretty easy to correct for. 

# ### Checking Column Values
# Before we standardize the column names, lets first make sure all of the values are similar.
# - if one file uses a very different format, we may want to correct it before combining

# In[3]:


from IPython.display import display

for df in [cohort1, cohort2, cohort3a, cohort3b]:
    display(df.head(3))


# They appear to match well, a few notes
# - `Unnamed: 0` in cohort3a sheet appears the same as Animal ID in the other dataframes
# - The first 2 cohorts do not record the cookie dought start/end, just the amount eaten

# ### Renaming Columns
# Let's now start to standardize the column names and combine them.
# - I use a lower case format and replace spaces with underscores. This fits well with python naming conventions and make them easy to access through pandas dataframes.

# In[4]:


columns_to_keep = ['id', 'latency_to_approach', 'time_in_corners', 'time_eating', 'amount_eaten']

cohort1 = cohort1.rename(columns={
    'Animal ID': 'id', 
    'Latency to approach': 'latency_to_approach', 
    'Time spent in corners': 'time_in_corners',
    'Time Spent Eating': 'time_eating',
    'Amount Eaten': 'amount_eaten'
})[columns_to_keep]

cohort2 = cohort2.rename(columns={
    'Animal ID': 'id', 
    'Latency to approach': 'latency_to_approach', 
    'Time spent in corners': 'time_in_corners',
    'Time Spent Eating': 'time_eating',
    'Amount Eaten': 'amount_eaten'
})[columns_to_keep]

cohort3a = cohort3a.rename(columns={
    'Unnamed: 0': 'id',
    'Latency': 'latency_to_approach', 
    'Corners': 'time_in_corners',
    'Eating ': 'time_eating',
    'Amount Eaten': 'amount_eaten'
})[columns_to_keep]

cohort3b = cohort3b.rename(columns={
    'Animal ID': 'id', 
    'Latency to Approach': 'latency_to_approach',
    'Time Spent in Corners': 'time_in_corners',
    'Time Spent Eating': 'time_eating',
    'Cookie Dough Eaten': 'amount_eaten'
})[columns_to_keep]


# In[5]:


for df in [cohort1, cohort2, cohort3a, cohort3b]:
    print(df.columns.tolist())


# Great! Now that out columns match up well, we can combine them.

# ## Combining Dataframes

# In[6]:


combined = pd.concat([cohort1, cohort2, cohort3a, cohort3b])
combined


# ## Removing trailing/leading spaces from strings
# Trailing/leading spaces are a common type of inconsistency in manually entered excel sheets
# 
# Let's remove any now

# In[7]:


def strip_strings(value):
    if isinstance(value, str):
        return value.strip()
    
    return value

combined = combined.applymap(strip_strings)


# ## Make id column lower case
# The video files that are named with the animal id use lower case letters, let's do the same here so it's easier to match them

# In[8]:


combined.id = combined.id.str.lower()
combined


# ## Removing text values

# In[9]:


for column in ['latency_to_approach', 'time_in_corners', 'time_eating', 'amount_eaten']:
    print(f"Column: {column}")
    non_numeric_values = combined[column].loc[combined[column].apply(type) == str].values
    print(non_numeric_values)
    print() # Add an empty row after printing out the column values


# Since there is some consistency to the text values, it seems useful to keep them. Let's move them to a new boolean column.

# In[10]:


import numpy as np

# Add new boolean columns for string information
combined["does_not_leave_corner"] = (
    combined.time_in_corners.str.contains("does not leave corner").fillna(False))

combined["does_not_approach"] = (
    combined.time_in_corners.str.contains("Does not approach").fillna(False))

# Remove the text values and just keep the numbers
# - We note they use 540 a few times when the mouse doesn't approach, let's continue to use that
combined = combined.replace("Does not approach", 540)
combined = combined.replace("Does not approach ", 540)
combined = combined.replace("540 (does not leave corner)", 540)


# *Side note: Sometimes `540` appears in the daya without the full text `'540 (does not leave corner)'`. This could have been data entry inconsistency, but we'll leave it as it's not clear.

# Let's confirm there are no strings left in the data

# In[11]:


for column in ['latency_to_approach', 'time_in_corners', 'time_eating', 'amount_eaten']:
    print(f"Column: {column}")
    non_numeric_values = combined[column].loc[combined[column].apply(type) == str].values
    print(non_numeric_values)
    print() # Add an empty row after printing out the column values


# ### Sorting the sheet

# Currently our sheet is not sorted

# In[12]:


combined.tail(3)


# Nope. We can sort it to make it a bit easier for humans to read

# In[13]:


def id_to_value(animal_id):
    """Sort the animals by sex then number"""
    sex = animal_id[0]
    number = int(animal_id[1:])
    sort_value = 1000 if sex == 'm' else 0
    sort_value += number
    return sort_value

combined = combined.sort_values('id', key=lambda x: combined.id.apply(id_to_value))


# In[14]:


combined


# ### Merging the treatment groups
# We have a separate csv sheet with the treatment data

# In[15]:


treatment_df = pd.read_csv(EXPERIMENT_DIR / "treatment-groups.csv")
treatment_df


# Let's check there aren't any typos with the values

# In[16]:


treatment_df.injected_with.unique()


# Looks good! Now let's merge it with the rest of our data

# In[17]:


combined = combined.merge(treatment_df, how='left', on='id')
combined


# # Saving the datasheet

# In[19]:


combined.to_csv(EXPERIMENT_DIR / 'experiment-data.csv', index=False)


# ## Done: What did we accomplish?
# - loaded data from several spreadsheets (4 for manual scores + 1 for treatment data)
# - removed inconsistencies
#   - replaced values
#   - removed tailing/leading spaces
#   - moved text values to new columns
# - made quality of life improvements
#   - renamed columns 
#   - sorted values 
# - merged datasheets and saved to a new file
