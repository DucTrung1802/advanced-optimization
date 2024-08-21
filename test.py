import pandas as pd


class ABC:
    def __init__(self, a, b):
        self.a = a
        self.b = b


df = pd.DataFrame(columns=["A", "B"])

my_abc = ABC("Trung", 24)

df.loc[len(df)] = [my_abc.a, my_abc.b]

print(df)
