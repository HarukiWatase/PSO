import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import rcParams, font_manager
import japanize_matplotlib

df = pd.read_csv('20241015_170126_rich.csv')
df.columns = ["iter", "PSO", "Dijkstra"]
df_plot = df.loc[:, 'PSO':'Dijkstra']

# ヒストグラム
plt.figure()
sns.set_theme(font='Times New Roman', font_scale=1.0, style="ticks")
df_melted = df_plot.melt(var_name='手法', value_name='ボトルネックリンク')
rcParams['font.size'] = 10.5
# y軸ラベルを日本語に変更
plt.ylabel('出現回数')  # 日本語に変更
# ヒストグラムのプロット
sns.histplot(data=df_melted, x='ボトルネックリンク', hue='手法', multiple='dodge', bins=np.arange(1,22), kde=False)
plt.savefig('comparison.pdf', bbox_inches='tight')

# 差分
plt.figure()
df['dif'] = df['PSO'] - df['Dijkstra']
sns.histplot(df['dif'], label='差分', color='green', kde=False)
plt.xlabel('ボトルネックリンク差')
plt.ylabel('出現回数')
plt.xticks(np.arange(-3, 21, 2))
plt.legend()
plt.savefig('dif.pdf', bbox_inches='tight')