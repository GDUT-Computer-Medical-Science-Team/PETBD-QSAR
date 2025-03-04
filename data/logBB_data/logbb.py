# Drop brain mean60min and blood mean60min columns from logBB1.csv
# and save the result as logbb.csv

import pandas as pd

# Read the original CSV file
df = pd.read_csv('./logBB1.csv')

# Keep only the columns we want (drop brain mean60min and blood mean60min)
df_new = df.drop(['brain mean60min', 'blood mean60min','Compound index'], axis=1)

# Save the new dataframe to logbb.csv
df_new.to_csv('./logBB.csv', index=False)