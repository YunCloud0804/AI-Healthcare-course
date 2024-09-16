import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joypy
import numpy as np

# Step 1: Load only relevant columns with chunksize to avoid MemoryError

# Step 2: Filter each chunk for relevant SUBJECT_IDs in the adm file and retain first chart time per subject
filtered_chunks = []
for chunk in chart_iter:
    chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
    chunk = chunk.dropna(subset=['CHARTTIME'])  # Drop rows with invalid dates
    chunk_filtered = chunk[chunk['SUBJECT_ID'].isin(adm['SUBJECT_ID'])]
    chunk_sorted = chunk_filtered.sort_values(by=['SUBJECT_ID', 'CHARTTIME'])
    chunk_first_time = chunk_sorted.drop_duplicates(subset='SUBJECT_ID', keep='first')
    filtered_chunks.append(chunk_first_time)

# Concatenate filtered chunks into a single dataframe
chart_first_time = pd.concat(filtered_chunks, ignore_index=True)

# Step 3: Merge with adm data to get ethnicity
merged_df = pd.merge(chart_first_time, adm, on='SUBJECT_ID', how='inner')

# Step 4: Downcast 'SUBJECT_ID' to int32 to reduce memory usage
merged_df['SUBJECT_ID'] = pd.to_numeric(merged_df['SUBJECT_ID'], downcast='integer')

# Step 5: Select top 6 ethnicities and convert to a list
top_ethnicities = merged_df['ETHNICITY'].value_counts().head(6).index.tolist()

# Step 6: Convert CHARTTIME to decimal time (e.g., 14:30 -> 14.5)
merged_df['TIME_OF_DAY'] = merged_df['CHARTTIME'].dt.hour + merged_df['CHARTTIME'].dt.minute / 60

# Step 7: Filter for the top 6 ethnicities
top_ethnicities_df = merged_df[merged_df['ETHNICITY'].isin(top_ethnicities)]

# Step 8: Create a data subset for plotting (only needed columns)
plot_data = top_ethnicities_df[['ETHNICITY', 'TIME_OF_DAY']]

# Step 9: Create a list of data for each ethnicity and filter out empty groups
data_by_ethnicity = [plot_data[plot_data['ETHNICITY'] == ethn]['TIME_OF_DAY'].dropna() for ethn in top_ethnicities]
data_by_ethnicity = [data for data in data_by_ethnicity if not data.empty]
filtered_ethnicities = [ethn for ethn, data in zip(top_ethnicities, data_by_ethnicity) if not data.empty]

# Set up color palette
colors = cm.viridis(np.linspace(0, 1, len(filtered_ethnicities)))

# Create the joyplot with 5-minute intervals
fig, axes = joypy.joyplot(data_by_ethnicity, labels=filtered_ethnicities, x_range=[0, 24],
                          bins=int(24 * 60 / 5),  # 24 hours, 5-minute bins
                          colormap=cm.viridis, figsize=(16, 9))

# Customize the plot
plt.xlabel('Time of Day (in 5-minute intervals)')
plt.ylabel('Ethnicity')
plt.title('Distribution of First Charttime for Top 6 Ethnicities')

# Show the plot
plt.tight_layout()
plt.savefig("chart-ethn.svg", format="svg")
plt.show()
