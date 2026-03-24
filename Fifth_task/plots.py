import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv('rng_validation.csv')

mt = df[df['Generator'] == 'MT19937']
bad = df[df['Generator'] == 'LowDiscrepancy']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
plt.subplots_adjust(hspace=0.4)

# --- ГРАФИК 1: Хи-квадрат ---
# Устанавливаем диапазон от 0 до 150, так как BadGen всегда 0, а MT около 100
ax1.hist(mt['Chi2'], bins=20, alpha=0.6, label='MT19937', color='orange')
ax1.hist(bad['Chi2'], bins=5, alpha=0.8, label='BadGenerator (Low Discrepancy)', color='blue')
ax1.set_xlim(-5, 150)
ax1.set_title('1. Chi-Squared Test (Uniformity in Bins)')
ax1.set_xlabel('Chi2 Statistic (Lower = More Uniform). BadGen is exactly 0.0!')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.2)

# --- ГРАФИК 2: Тест Колмогорова-Смирнова ---
# Твоему генератору так хорошо, что зумимся в диапазон 0 - 0.03
ax2.hist(mt['KS'], bins=20, alpha=0.6, label='MT19937', color='orange')
ax2.hist(bad['KS'], bins=20, alpha=0.8, label='BadGenerator (Low Discrepancy)', color='blue')
ax2.set_xlim(0, 0.03)
ax2.set_title('2. Kolmogorov-Smirnov Test (CDF Distance)')
ax2.set_xlabel('KS Statistic (Lower is better)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.2)

# --- ГРАФИК 3: Автокорреляция ---
# Здесь виден провал: MT около 0, BadGen около 1.0
ax3.hist(mt['AutoCorr'], bins=20, alpha=0.6, label='MT19937', color='orange')
ax3.hist(bad['AutoCorr'], bins=5, alpha=0.8, label='BadGenerator (Low Discrepancy)', color='blue')
ax3.axvline(x=0, color='red', linestyle='--', label='Ideal Random')
ax3.set_xlim(-0.1, 1.2)
ax3.set_title('3. Autocorrelation Test (Independence)')
ax3.set_xlabel('Correlation Value (0.0 = Independent, 1.0 = Deterministic)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.2)

plt.savefig('rng_final_report.png', dpi=300, bbox_inches='tight')
plt.show()