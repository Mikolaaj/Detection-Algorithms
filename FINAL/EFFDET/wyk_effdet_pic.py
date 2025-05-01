import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Ścieżki do plików
file_paths = [
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d0/d0_pic.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d1/d1_pic.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d2/d2_pic.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d3/d3_pic.csv',
    '/mnt/d/home/miko/Yet-Another-EfficientDet-Pytorch/WYNIKI/CATTLE/effdet_d4/d4_pic.csv'
]

# Kolory
colors = cm.tab10.colors

# WYKRES mAP
fig_map, ax_map = plt.subplots(figsize=(12, 5))
ax_map.set_title('Współczynnik Prawdopodobieństwa mAP dla EfficientDet na RTX')
ax_map.set_xlabel('Numer zdjęcia')
ax_map.set_ylabel('Współczynnik mAP')

# WYKRES opóźnień
fig_delay, ax_delay = plt.subplots(figsize=(12, 5))
ax_delay.set_title('Wykres opóźnienia ms dla EfficientDet na RTX')
ax_delay.set_xlabel('Numer zdjęcia')
ax_delay.set_ylabel('Opóźnienie [ms]')

# Przetwarzanie każdego pliku
for i, filepath in enumerate(file_paths):
    df = pd.read_csv(filepath, skipinitialspace=True)
    print(f"===> Kolumny w {filepath}: {df.columns.tolist()}")
    color = colors[i % len(colors)]
    file_label = f'd{i}'

    # mAP
    ax_map.plot(df['nr_zdjecia'].to_numpy(), df['mAP'].to_numpy(), marker='o', color=color, label=f'{file_label} mAP')
    min_map = df['mAP'].min()
    max_map = df['mAP'].max()
    ax_map.axhline(min_map, color=color, linestyle='--', linewidth=1, label=f'{file_label} mAP min={min_map:.2f}')
    ax_map.axhline(max_map, color=color, linestyle=':', linewidth=1, label=f'{file_label} mAP max={max_map:.2f}')

    # Opóźnienie (pomijamy pierwsze 2 wiersze)
    df_delay = df.iloc[2:]
    ax_delay.plot(df_delay['nr_zdjecia'].to_numpy(), df_delay['opoznienie_ms'].to_numpy(), marker='o', color=color, label=f'{file_label} opóźnienie')
    min_delay = df_delay['opoznienie_ms'].min()
    max_delay = df_delay['opoznienie_ms'].max()
    ax_delay.axhline(min_delay, color=color, linestyle='--', linewidth=1, label=f'{file_label} opóźnienie min={min_delay:.0f}')
    ax_delay.axhline(max_delay, color=color, linestyle=':', linewidth=1, label=f'{file_label} opóźnienie max={max_delay:.0f}')

# Legendy i siatki
ax_map.legend(loc='lower right')
ax_map.grid(True)

ax_delay.legend(loc='lower right')
ax_delay.grid(True)

# Zapis wykresów
fig_map.tight_layout()
fig_map.savefig("wykres_map_pic.png")
plt.close(fig_map)

fig_delay.tight_layout()
fig_delay.savefig("wykres_opoznienie_pic.png")
plt.close(fig_delay)
