import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SAVE_PLOTS = True  # Set to False to display plots interactively instead of saving

df = pd.read_csv('diamonds.csv')

#usunięcie x,y,z = 0
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

#region wstępne wykresy
if SAVE_PLOTS:
    output_folder = 'wykresy'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols[1:]:
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

    summary = df[numeric_cols[1:]].agg(['min', 'max', 'mean']).transpose()
    fig, ax = plt.subplots(figsize=(10, len(numeric_cols) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary.values,
                     colLabels=summary.columns,
                     rowLabels=summary.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Podsumowanie statystyczne (min, max, średnia)', fontweight="bold")

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



features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']

# Skalowanie danych (Standaryzacja: średnia = 0, odchylenie standardowe = 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Przekształcamy do DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=features)



print(df_scaled.head())  # Podgląd pierwszych 5 wierszy
print(df_scaled.info())  # Sprawdzenie typów danych i braków
print(df_scaled.describe())  # Statystyki opisowe



wcss = []  # Suma błędów wewnątrz klastrów (Within-Cluster Sum of Squares)

# Testujemy od 1 do 10 klastrów
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ to suma błędów w klastrach

# Wykres metody łokcia
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('WCSS')
plt.title('Metoda łokcia')
plt.show()
