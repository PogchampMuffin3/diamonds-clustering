import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

GENERATE_CHARTS = True
COUNT_CLUSTERS = True
ANALYSIS_METHODS = True

df = pd.read_csv('diamonds.csv')
output_folder = 'output'

#usunięcie x,y,z = 0
original_count = df.shape[0]
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
removed = original_count-df.shape[0]
print(f"Usunieto {removed} wierszy")
print(f"Usunieto {removed/original_count*100}% wierszy")

#region wstępne wykresy
if GENERATE_CHARTS:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[1:]
    categorical_cols = df.select_dtypes(include=['str', 'category']).columns

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax.grid(axis='y', linestyle='-', alpha=0.6)
        ax.set_title(f'Histogram: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Liczba')
        plt.tight_layout()

        fig.savefig(os.path.join(output_folder, f'histogram_{col}.png'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        box = ax.boxplot(df[col].dropna(), patch_artist=True,
                         boxprops=dict(facecolor="skyblue", color="black"),
                         flierprops=dict(marker='o', markersize=2, color='black', alpha=0.5))

        median = df[col].median()
        mean = df[col].mean()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        ax.axhline(median, color='red', linestyle='--', label='Mediana')  # Mediana
        ax.axhline(mean, color='green', linestyle='-', alpha=0.6, label='Średnia')  # Średnia
        ax.axhline(q1, color='purple', linestyle='dotted', alpha=0.7, label='Q1 (25%)')  # 1. kwartyl
        ax.axhline(q3, color='brown', linestyle='dotted', alpha=0.7, label='Q3 (75%)')  # 3. kwartyl
        ax.grid(axis='y', linestyle='-', alpha=0.6)

        ax.set_title(f'Boxplot: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Wartości')
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, f'boxplot_{col}.png'))
        plt.close(fig)

    for col in categorical_cols:
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Wykres kołowy: {col}')
        plt.tight_layout()

        fig.savefig(os.path.join(output_folder, f'pie_{col}.png'))
        plt.close(fig)

    summary = df[numeric_cols].agg(['min', 'max', 'mean', 'median', 'std', 'skew']).transpose().round(8)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary.values,
                     colLabels=summary.columns,
                     rowLabels=summary.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Podsumowanie statystyczne (min, max, średnia, mediana, odchylenie standardowe, skośność)', fontweight="bold")

    fig.savefig(os.path.join(output_folder, 'statystyki_podsumowanie.png'))
    plt.close(fig)

#endregion

#region zamiana jakościowych na ilościowe
cut_mapping = {
    'Ideal': 1,
    'Premium': 2,
    'Very Good': 3,
    'Good': 4,
    'Fair': 5
}
color_mapping = {
    'D': 1,
    'E': 2,
    'F': 3,
    'G': 4,
    'H': 5,
    'I': 6,
    'J': 7
}
clarity_mapping = {
    'IF': 1,
    'VVS1': 2,
    'VVS2': 3,
    'VS1': 4,
    'VS2': 5,
    'SI1': 6,
    'SI2': 7,
    'I1': 8
}

df['cut'] = df['cut'].map(cut_mapping)
df['color'] = df['color'].map(color_mapping)
df['clarity'] = df['clarity'].map(clarity_mapping)
#endregion

#dodanie przedziałów cenowych (kwantyle)
df['price_category'] = pd.qcut(df['price'], q=3, labels=[3,2,1])
df['index'] = df.index

#region skalowanie
features = ['carat', 'cut', 'color', 'clarity', 'x', 'y', 'z', 'depth', 'table']
df_cluster = df[features].copy()

scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)
WARD_IDX = np.random.RandomState(42).choice(len(df_cluster_scaled), 10000, replace=False)
ward_data = df_cluster_scaled[WARD_IDX]
#endregion

#region obliczanie optymalnej ilości klastrów
if COUNT_CLUSTERS:
    #KMeans - łokieć
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_cluster_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('KMeans – Metoda łokcia')
    plt.xlabel('Liczba klastrów (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'KMeans_lok.png'))
    plt.close()

    #KMeans - silhouette score
    silhouette_scores_kmeans = []
    for k in range(2, 11):
        print(f'Licze: {k}')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_cluster_scaled)
        score = silhouette_score(df_cluster_scaled, labels, sample_size=10000, random_state=42)
        silhouette_scores_kmeans.append(score)
        print(f'KMeans - Liczba klastrów: {k}, Silhouette Score: {score:.4f}')

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 11), silhouette_scores_kmeans, marker='o')
    plt.title('KMeans – Silhouette Score')
    plt.xlabel('Liczba klastrów (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'KMeans_sil.png'))
    plt.close()

    #Ward - silhouette score
    silhouette_scores_ward = []
    for k in range(2, 11):
        print(f'Licze: {k}')
        ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = ward.fit_predict(ward_data)
        score = silhouette_score(ward_data, labels)
        silhouette_scores_ward.append(score)
        print(f'Ward - Liczba klastrów: {k}, Silhouette Score: {score:.4f}')

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 11), silhouette_scores_ward, marker='o')
    plt.title('Ward – Silhouette Score')
    plt.xlabel('Liczba klastrów (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'Ward_sil.png'))
    plt.close()
#endregion

#region klasteryzacja
if not ANALYSIS_METHODS: exit(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(df_cluster_scaled)

ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
ward_labels_sample = ward.fit_predict(ward_data)
df_ward = df.iloc[WARD_IDX].copy()
df_ward['cluster_ward'] = ward_labels_sample

cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
df['kmeans_color'] = df['cluster_kmeans'].map(cluster_colors)
cluster_colors = {0: 'blue', 1: 'red', 2: 'green'}
df_ward['ward_color'] = df_ward['cluster_ward'].map(cluster_colors)
cluster_colors = {1: 'green', 2: 'blue', 3: 'red'}
df['price_color'] = df['price_category'].map(cluster_colors)
df_ward['price_color'] = df_ward['price_category'].map(cluster_colors)
#endregion

#region obliczanie pokrycia
df['kmeans_vs_cena_match'] = df['kmeans_color'] == df['price_color']
df_ward['ward_vs_cena_match'] = df_ward['ward_color'] == df_ward['price_color']
print(f"Pokrycie KMeans: {df['kmeans_vs_cena_match'].mean()*100}")
print(f"Pokrycie Ward (próbka 10k): {df_ward['ward_vs_cena_match'].mean()*100}")
#endregion

#region wizualizacja
def make_plots(x,y, ward_df=None):    #x i y to nazywy kolumn
    # KMeans - pełny zbiór
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df[x], y=df[y], hue=df['cluster_kmeans'], palette=cluster_colors, alpha=0.1, edgecolors='none')
    plt.title(f'K-Means: Klasyfikacja diamentów na podstawie {x} vs {y}')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(output_folder, f'K-Means_{x}_{y}.png'))
    plt.close()

    # Ward — próbka 10k
    if ward_df is not None:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=ward_df[x], y=ward_df[y], hue=ward_df['cluster_ward'], palette=cluster_colors, alpha=0.1, edgecolors='none')
        plt.title(f'Ward: Klasyfikacja diamentów na podstawie {x} vs {y} (próbka 10k)')
        plt.legend(title='Cluster')
        plt.savefig(os.path.join(output_folder, f'Ward_{x}_{y}.png'))
        plt.close()

    cluster_colors = {1: 'green', 2: 'blue', 3: 'red'}
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df[x], y=df[y], hue=df['price_category'], palette=cluster_colors, alpha=0.1, edgecolors='none')
    plt.title(f'Grupowanie diamentów na podstawie {x} vs {y}')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(output_folder, f'Wykresy_borderless{x}_{y}.png'))
    plt.close()

for f in numeric_cols:
    for s in numeric_cols:
        if f==s: continue
        make_plots(f,s, ward_df=df_ward)

#endregion