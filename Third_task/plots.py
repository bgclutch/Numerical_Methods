import pandas as pd
import matplotlib.pyplot as plt

filename = 'precision_results.csv'
df = pd.read_csv(filename)
df.columns = df.columns.str.strip()

f32_df = df[df['Type'] == 'float32'].reset_index()
f64_df = df[df['Type'] == 'float64'].reset_index()

x_axis = f32_df['Number'] if 'Number' in f32_df.columns else f32_df.index + 1

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

# Крах алгоритма Naive на float32
ax1.set_yscale('log')
ax1.plot(x_axis, f32_df['ErrFast'], 'ro-', label='Naive (Fast)', linewidth=2)
ax1.plot(x_axis, f32_df['ErrTwoPass'], 'bo-', label='Two-Pass (Stable)', alpha=0.6)
ax1.plot(x_axis, f32_df['ErrSinglePass'], 'go-', label='Welford (Single-Pass)', linewidth=2)

# Добавляем линию порога 100% ошибки
ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
ax1.text(1, 1.2, '100% Relative Error (Total Failure)', color='black', fontweight='bold')

ax1.set_title('Task 4: Algorithm Stability Analysis (float32)', fontsize=14)
ax1.set_xlabel('Test Case (Increasing Mean/StdDev ratio)')
ax1.set_ylabel('Relative Error (Log Scale)')
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend()

# Сравнение точности float32 vs float64 для метода Naive
ax2.set_yscale('log')
ax2.plot(x_axis, f32_df['ErrFast'], 'r--', label='Naive @ float32', linewidth=2)
ax2.plot(x_axis, f64_df['ErrFast'], 'b-', label='Naive @ float64 (Double)', linewidth=2, marker='s')

ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
ax2.text(1, 1.2, '100% Relative Error (Total Failure)', color='black', fontweight='bold')

ax2.set_title('Task 4: Hardware Precision Impact (Naive Method)', fontsize=14)
ax2.set_xlabel('Test Case')
ax2.set_ylabel('Relative Error (Log Scale)')
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend()

# float64
ax3.set_yscale('log')
ax3.plot(x_axis, f64_df['ErrFast'], 'ro-', label='Naive (Fast)', linewidth=2)
ax3.plot(x_axis, f64_df['ErrTwoPass'], 'bo-', label='Two-Pass (Stable)', alpha=0.7)
ax3.plot(x_axis, f64_df['ErrSinglePass'], 'go-', label='Welford (Single-Pass)', linewidth=2)

ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
ax3.text(1, 1.2, '100% Relative Error (Total Failure)', color='black', fontweight='bold')

ax3.set_title('Variance Calculation Precision: float64 (Double)', fontsize=14)
ax3.set_xlabel('Test Case')
ax3.set_ylabel('Relative Error (Log Scale)')
ax3.grid(True, which="both", ls="-", alpha=0.2)
ax3.legend()


plt.savefig('critical_error.png', dpi=300)
plt.show()