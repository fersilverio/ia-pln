import pandas as pd

up = lambda s: s.upper()


array = ['a', 'b', 'c']

df_array = pd.DataFrame(data=array)

print(df_array)

# Atribuindo modificação para um novo df
a = df_array.applymap(up)

print(a)

#Acessando elemento no dataframe
print(a[0][2])

