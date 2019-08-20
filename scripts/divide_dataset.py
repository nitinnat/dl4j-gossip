import pandas as pd
import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--filepath",  help="Path to csv",
	    type = str,required =True)
	parser.add_argument("--num_splits" , help="Number of splits",
	    type = int,required =True)
	args = parser.parse_args()
	data = pd.read_csv(args.filepath, header=None)
	num_samples = len(data)
	samples_per_dataframe = num_samples//args.num_splits
	folder = os.path.dirname(args.filepath)
	print(folder)
	for i in range(args.num_splits-1):
		temp_df = data.loc[i*samples_per_dataframe:(i+1)*samples_per_dataframe]
		temp_df.to_csv(folder, "t_" + str(i) + ".csv")

