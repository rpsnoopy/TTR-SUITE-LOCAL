import pandas as pd
df = pd.read_csv(r'C:\TTR_Benchmark\results\raw_results_20260223_134841.csv')
cuad = df[df['benchmark']=='cuad']
print('CUAD rows:', len(cuad))
ic = cuad['is_correct'].astype(float)
print('mean F1:', round(ic.mean(), 3))
print('non-zero:', (ic > 0).sum())
print('zero:', (ic == 0).sum())
print()
print(cuad[['task_id','is_correct','response']].head(10).to_string())
