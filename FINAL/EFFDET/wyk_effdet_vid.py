import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from matplotlib.lines import Line2D

# Ścieżki do plików CSV
file_paths = [
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d0/d0_vid.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d1/d1_vid.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d2/d2_vid.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d3/d3_vid.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d4/d4_vid.csv'
]

# Ustawienia kolorów
colors = cm.tab10.colors

# ============ WYKRES 1: mAP ============

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_title('Współczynnik Prawdopodobieństwa mAP EfficientDet na RTX')
ax1.set_xlabel('Numer klatki')
ax1.set_ylabel('Współczynnik mAP')
ax1.grid(True)

legend_elements_map = []

# ============ WYKRES 2: Opóźnienie ============

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.set_title('Opóźnienie [ms] EfficientDet na RTX')
ax2.set_xlabel('Numer klatki')
ax2.set_ylabel('Opóźnienie [ms]')
ax2.grid(True)

legend_elements_delay = []

# Funkcja do usuwania outlierów (metoda IQR)
def remove_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]

# Zbiorcze dane do uśredniania
map_dfs = []
delay_dfs = []

# Iteracja po plikach CSV
for i, filepath in enumerate(file_paths):
    df = pd.read_csv(filepath)
    file_label = f'd{i}'
    color = colors[i % len(colors)]

    # === mAP (bez zer)
    mAP_filtered = remove_outliers(df['mAP'][(df['mAP'] > 0)])
    df_map = df[df['mAP'].isin(mAP_filtered)]
    ax1.plot(df_map['nr_klatki'].to_numpy(), df_map['mAP'].to_numpy(), marker='o', color=color)
    min_map = df_map['mAP'].min()
    max_map = df_map['mAP'].max()
    ax1.axhline(min_map, color=color, linestyle='--', linewidth=1)
    ax1.axhline(max_map, color=color, linestyle='--', linewidth=1)
    legend_elements_map.append(Line2D([0], [0], color=color, label=f'{file_label}'))
    legend_elements_map.append(Line2D([0], [0], color=color, linestyle='--', label=f'{file_label} min={min_map:.2f}'))
    legend_elements_map.append(Line2D([0], [0], color=color, linestyle='--', label=f'{file_label} max={max_map:.2f}'))

    # Dodaj do listy do uśredniania mAP
    map_dfs.append(df_map[['nr_klatki', 'mAP']])

    # === Opóźnienie
    df_filtered = df[df['nr_klatki'] >= 10]
    opoznienia_filtered = remove_outliers(df_filtered['opoznienie_ms'])
    df_delay = df_filtered[df_filtered['opoznienie_ms'].isin(opoznienia_filtered)]
    ax2.plot(df_delay['nr_klatki'].to_numpy(), df_delay['opoznienie_ms'].to_numpy(), marker='o', color=color)
    min_delay = df_delay['opoznienie_ms'].min()
    max_delay = df_delay['opoznienie_ms'].max()
    ax2.axhline(min_delay, color=color, linestyle='--', linewidth=1)
    ax2.axhline(max_delay, color=color, linestyle='--', linewidth=1)
    legend_elements_delay.append(Line2D([0], [0], color=color, label=f'{file_label}'))
    legend_elements_delay.append(Line2D([0], [0], color=color, linestyle='--', label=f'{file_label} min={min_delay:.0f} ms'))
    legend_elements_delay.append(Line2D([0], [0], color=color, linestyle='--', label=f'{file_label} max={max_delay:.0f} ms'))

    # Dodaj do listy do uśredniania opóźnień
    delay_dfs.append(df_delay[['nr_klatki', 'opoznienie_ms']])

# Dodanie legend
ax1.legend(handles=legend_elements_map, loc='lower right', fontsize='small')
ax2.legend(handles=legend_elements_delay, loc='lower right', fontsize='small')

# ============ WYKRES 3: Średni mAP ============

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.set_title('Średni współczynnik mAP EfficientDet na RTX')
ax3.set_xlabel('Numer klatki')
ax3.set_ylabel('Średni mAP')
ax3.grid(True)

for i, df in enumerate(map_dfs):
    mean_map = df.groupby('nr_klatki').mean().reset_index()
    
    # Wybór klatek: min, max, + 4 równomierne
    min_row = mean_map.loc[mean_map['mAP'].idxmin()]
    max_row = mean_map.loc[mean_map['mAP'].idxmax()]
    
    sample_rows = mean_map.sort_values('nr_klatki').iloc[::max(1, len(mean_map)//6)]
    selected = pd.concat([min_row.to_frame().T, max_row.to_frame().T, sample_rows]).drop_duplicates().sort_values('nr_klatki')
    
    # Konwersja na tablice NumPy przed rysowaniem wykresu
    ax3.plot(selected['nr_klatki'].to_numpy(), selected['mAP'].to_numpy(), marker='o', label=f'd{i}', color=colors[i % len(colors)] )

    # Dodajemy poziome linie na wysokości min i max wartości mAP
    min_mAP = min_row['mAP']
    max_mAP = max_row['mAP']
    ax3.axhline(min_mAP, color=colors[i % len(colors)], linestyle='--', label=f'{file_label} min={min_mAP:.2f}')
    ax3.axhline(max_mAP, color=colors[i % len(colors)], linestyle='--', label=f'{file_label} max={max_mAP:.2f}')

ax3.legend(title='Model', loc='lower right')


# ============ WYKRES 4: Średnie opóźnienie ============

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.set_title('Średnie opóźnienie [ms] EfficientDet na RTX')
ax4.set_xlabel('Numer klatki')
ax4.set_ylabel('Średnie opóźnienie [ms]')
ax4.grid(True)

for i, df in enumerate(delay_dfs):
    mean_delay = df.groupby('nr_klatki').mean().reset_index()
    
    min_row = mean_delay.loc[mean_delay['opoznienie_ms'].idxmin()]
    max_row = mean_delay.loc[mean_delay['opoznienie_ms'].idxmax()]
    
    sample_rows = mean_delay.sort_values('nr_klatki').iloc[::max(1, len(mean_delay)//6)]
    selected = pd.concat([min_row.to_frame().T, max_row.to_frame().T, sample_rows]).drop_duplicates().sort_values('nr_klatki')
    
    # Konwersja na tablice NumPy przed rysowaniem wykresu
    ax4.plot(selected['nr_klatki'].to_numpy(), selected['opoznienie_ms'].to_numpy(), marker='o', label=f'd{i}', color=colors[i % len(colors)])

    # Dodajemy poziome linie na wysokości min i max wartości opóźnienia
    min_delay = min_row['opoznienie_ms']
    max_delay = max_row['opoznienie_ms']
    ax4.axhline(min_delay, color=colors[i % len(colors)], linestyle='--', label=f'{file_label} min={min_delay:.0f} ms')
    ax4.axhline(max_delay, color=colors[i % len(colors)], linestyle='--', label=f'{file_label} max={max_delay:.0f} ms')

ax4.legend(title='Model', loc='lower right')


# Zapis wykresów
plt.tight_layout()
fig1.savefig('map_wszystkie.png')
fig2.savefig('opoznienie_wszystkie.png')
fig3.savefig('map_sredni.png')
fig4.savefig('opoznienie_srednie.png')

# Podsumowanie
print("✅ Wygenerowano wszystkie wykresy:")
print("   - 'map_wszystkie.png' (wszystkie mAP)")
print("   - 'opoznienie_wszystkie.png' (wszystkie opóźnienia)")
print("   - 'map_sredni.png' (średni mAP)")
print("   - 'opoznienie_srednie.png' (średnie opóźnienie)")
