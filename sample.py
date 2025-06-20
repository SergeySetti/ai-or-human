import pandas as pd

df = pd.read_csv("data/AI_Human.csv", engine='python')

per_target = 7000

sample_df = df.groupby('generated').apply(lambda x: x.sample(per_target, random_state=77)).reset_index(drop=True)

print(sample_df['generated'].value_counts())

sample_df.to_parquet(f"data/sample_{per_target}_per_target.parquet", index=False, engine='pyarrow')
