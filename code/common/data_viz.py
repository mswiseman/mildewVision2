import pandas as pd
import seaborn
import sns as sns
import matplotlib.pyplot as plt
from seaborn import set_style
from seaborn import catplot

# Load the provided CSV file
file_path = "D:\\11_22_2023_grape_baseline_concatenated_results.csv"
data = pd.read_csv(file_path)

data['imaging_date'] = data['imaging_date'].astype(str)
filtered_data = data[data['imaging_date'].str.endswith('_5dpi')]

# Preparing the data: Grouping by 'name' and 'model_timestamp'
grouped_data = filtered_data.groupby(['model_timestamp', 'name'])

# Generating the plots
g = seaborn.catplot(x="name", y="severity_rate_patch", col="model_timestamp",
                data=filtered_data, kind="box", col_wrap=2, height=5, aspect=2)

# Rotating x labels for better readability
g.set_xticklabels(rotation=90)

# Setting titles and labels
g.set_titles("{col_name}")
g.set_axis_labels("Sample Name", "Severity Rate")

plt.subplots_adjust(top=0.9)
g.fig.suptitle('Box and Whisker Plot for Each Name x Model Time Combination, Faceted by Model Date')

plt.show()