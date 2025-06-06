# visualize.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairwise(csv_path, title_prefix):
    df = pd.read_csv("public_data.csv")
    df = df.iloc[:, [2, 3, 4]]  # drop id and S1

    sns.set(style="whitegrid", context="notebook")

    # Pairplot with kernel density on the diagonal
    pairplot = sns.pairplot(df, kind="scatter", plot_kws={"s": 5}, diag_kind="kde")

    pairplot.fig.suptitle(f"{title_prefix} - Pairwise Scatter", y=1.02)
    plt.show()

if __name__ == "__main__":
    plot_pairwise("public_data.csv", title_prefix="Public Data")