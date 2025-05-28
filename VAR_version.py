from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from spectral_connectivity import Multitaper, Connectivity
from matplotlib.colors import LogNorm, Normalize     # ← добавлено
from scipy.signal import butter, filtfilt

start_scope()

# Параметры моделирования
sampling_frequency = 100  # Гц (100 - чтобы увидеть частоты 0-40, 200 чтобы увидеть частоты 0-80)
dt_sim = 1 / sampling_frequency  # шаг интегрирования (сек)
defaultclock.dt = dt_sim * second
noise_std = 0.1     # стандартное отклонение шума

# Временные параметры
t_total_sim = 10   # общее время моделирования (сек)
segment_switch = 5  # время переключения драйвера (сек)

n_neurons = 100
fs = 100
n_samples = int(fs * t_total_sim)
A = 200 * pA
B = 25 * pA
R = 80 * Mohm
f = 10*Hz  
f2 = 30*Hz     
tau = 20*ms
phi = 0
J = 1 * mV
eqs = '''
dv/dt = (v_rest-v+R*(I_half1+I_half2))/tau :  volt            
I_half1 = int(t < 5000*ms) * amplitude * sin(2*pi*f*t + phi) : amp              
I_half2 = int(t >= 5000*ms) * amplitude2 * sin(2*pi*f2*t + phi) : amp
amplitude : amp 
amplitude2 : amp 
'''
v_threshold = -50 * mV
v_reset = -70 * mV
v_rest =  -65 * mV

G = NeuronGroup(n_neurons, 
                eqs, 
                threshold="v > v_threshold",
                reset="v = v_reset",
                method='euler')
G.v = v_rest
# Задаем различные амплитуды для двух подгрупп нейронов
G.amplitude[:n_neurons//2] = A  # нейроны с 0 по 24 получают амплитуду A
G.amplitude[n_neurons//2:] = B  # нейроны с 25 по 50 получают амплитуду B
G.amplitude2[:n_neurons//2] = B  # нейроны с 25 по 50 получают амплитуду B
G.amplitude2[n_neurons//2:] = A  # нейроны с 0 по 24 получают амплитуду A

p_intra1 = 0.15
p_intra2 = 0.15
p_12     = 0.05
p_21     = 0.05

# Веса соединения
w_intra1 = 6.6
w_intra2 = 6.6
w_12     = 10
w_21     = 1

n_half = n_neurons // 2

input_rate = 10 * Hz
input_group = PoissonGroup(n_neurons, rates=input_rate)
syn_input = Synapses(input_group, G, on_pre='v_post += J')
syn_input.connect(p=0.2)

S_intra1 = Synapses(G, G, model='w : 1', on_pre='v_post += J * w')
S_intra1.connect(
    condition='i <= n_half and j <= n_half',
    p=p_intra1
)
S_intra1.w = w_intra1

# 2) Синапсы внутри 2-го кластера
S_intra2 = Synapses(G, G, model='w : 1', on_pre='v_post += J * w')
S_intra2.connect(
    condition='i >= n_half and j >= n_half',
    p=p_intra2
)
S_intra2.w = w_intra2

# 3) Синапсы из 1-го кластера во 2-й
S_12 = Synapses(G, G, model='w : 1', on_pre='v_post += J * w')
S_12.connect(
    condition='i < n_half and j >= n_half',
    p=p_12
)
S_12.w = w_12

# 4) Синапсы из 2-го кластера в 1-й
S_21 = Synapses(G, G, model='w : 1', on_pre='v_post += J * w')
S_21.connect(
    condition='i >= n_half and j < n_half',
    p=p_21
)
S_21.w = w_21


W = np.zeros((n_neurons, n_neurons))
W[S_intra1.i[:], S_intra1.j[:]] = S_intra1.w[:]
W[S_intra2.i[:], S_intra2.j[:]] = S_intra2.w[:]
W[S_12.i[:], S_12.j[:]] = S_12.w[:]
W[S_21.i[:], S_21.j[:]] = S_21.w[:]
plt.matshow(W, cmap='viridis')

mon = StateMonitor(G, 'v', record=True)
spike_monitor = SpikeMonitor(G)

# Запуск моделирования
run(t_total_sim * second)

spike_times = spike_monitor.t / second
spike_indices = spike_monitor.i
plt.figure(figsize=(10, 8))
plt.scatter(spike_times, spike_indices, marker='|')
plt.xlim(0,10)
plt.ylim(0,100)
plt.xlabel('Время (сек)', fontsize=14)
plt.ylabel('Нейроны', fontsize=14)

# Извлечение данных
t_sim = mon.t/second
x1 = mon.v[:n_neurons//2, :] / mV  # (форма: n_neurons//2, 1000)
x2 = mon.v[n_neurons//2:, :] / mV  # (форма: n_neurons//2, 1000)


v1 = mon.v[:n_neurons//2 , :].mean(axis=0) / mV             # кластер 1
v2 = mon.v[n_neurons//2:, :].mean(axis=0) / mV              # кластер 2
v3 = mon.v[:, :].mean(axis=0) / mV              # кластер 2


trial0 = x1.T  # (1000, n_neurons//2)
trial1 = x2.T  # (1000, n_neurons//2)
signal = np.stack((trial0, trial1), axis=-1)  # (1000, n_neurons//2, 2)

# Добавление фонового шума
signal_noisy = signal + np.random.normal(0, 1, signal.shape)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t_sim, signal_noisy[:,0,0], label="Сигнал 1")
axes[0].set_xlabel("Время (сек)", fontsize=14)
axes[0].set_ylabel("Мембранный потенциал (мВ)", fontsize=14)
axes[0].legend()

axes[1].plot(t_sim, signal_noisy[:,0,1], label="Сигнал 2")
axes[1].set_xlabel("Время (сек)", fontsize=14)
axes[1].set_ylabel("Мембранный потенциал (мВ)", fontsize=14)
axes[1].legend()

v  = np.stack([v1, v2], axis=-1)
v_noisy = v + np.random.normal(0, 1, v.shape)

print(v_noisy.shape)


win_len, step, order = 256, 64, 4
nfft      = 256
freqs     = np.linspace(0, fs/2, nfft//2 + 1) # 0-50
n_freq    = len(freqs) # 129

def spectra_2x2(A, Sigma, i, j):
    gc  = np.empty(n_freq)
    dtf = np.empty(n_freq)
    pdc = np.empty(n_freq)

    for l, f in enumerate(freqs): # l=129 f=0-50
        z  = np.exp(-1j * 2*np.pi * f / fs)
        Az = np.eye(2, dtype=complex) # [[1+j, 0+j],[0+j, 1+j]]
        for p, Al in enumerate(A, start=1): # p=4 Al=(2,2)
            Az -= Al * z**p

        Hz = np.linalg.inv(Az) # Az=[[a,b][c,d]] A(z)^{-1}=1/(ad-bc)[[d, -b],[-c, a]]
        Sz = Hz @ Sigma @ Hz.conj().T # S(z)=H(z)ΣH^*(z) Hz=[[3+j, 5],[2-j, j]] Hz^*=[[3-j, 2+j],[5, -j]]
        S_cond = Sz[i, i] - Sz[i, j] * (1.0 / Sz[j, j]) * Sz[j,i]  # S_ii|j = S_ii − S_ij*S_jj^{-1}*S_ji
        gc[l]  = np.log(np.real(Sz[i, i] / S_cond)) # ln(S_11(w)/S_(11|2)(w)) np.real - берем действительную часть числа (без j)
        dtf[l] = np.abs(Hz[i, j])**2 / (np.abs(Hz[i, 0])**2 + np.abs(Hz[i, 1])**2)  # |Hz_ij|^2/(|Hz_i0|^2+|Hz_i1|^2)
        pdc[l] = np.abs(Az[i, j])**2 / (np.abs(Az[0, j])**2 + np.abs(Az[1, j])**2)  # |Az_ij|^2/(|Az_0j|^2+|Az_1j|^2)
    return gc, dtf, pdc

def var_ols_const(y, p=4):
    y = np.asarray(y, float)
    T, k = y.shape # (256, 2)
    Y = y[p:] # (252, 2)
    lags = [y[p-l-1:T-l-1] for l in range(p)] # (4, 252, 2) -> смещенные сигналы
    X = np.hstack([np.ones((T-p, 1)), *lags]) # (252, 9) -> N=T-p=252, 1+k*p=1+2*4=9 -> массив единиц

    # β = (XᵀX)⁻¹XᵀY
    XtX = X.T @ X # (9, 9)
    XtY = X.T @ Y # (9, 2)
    B = np.linalg.solve(XtX, XtY) # (9, 2) решаем Ax=b, где XtX-A, XtY-b
    B_new = [B[1+i*k : 1+(i+1)*k].T for i in range(p)] # (4, 2, 2)
    A = np.stack(B_new, axis=0) # (4, 2, 2) -> собираем в один тензор

    E = Y - X @ B # (252, 2)
    dof = (T-p) - (k*p + 1) # 243
    Sigma = (E.T @ E) / dof # (2, 2)
    return A, Sigma # (4, 2, 2) и (2, 2)

def slide_2x2(i, j):
    G, D, P, T = [], [], [], []
    for start in range(0, n_samples - win_len + 1, step): # 64 (0-745)
        seg = v_noisy[start:start + win_len]
        A, Sigma = var_ols_const(seg, order)
        g, d, p = spectra_2x2(A, Sigma, i, j)
        G.append(g) 
        D.append(d) 
        P.append(p)
        T.append((start + win_len//2) / fs)
    return np.array(G).T, np.array(D).T, np.array(P).T, np.array(T)


gc_12, dtf_12, pdc_12, times = slide_2x2(0, 1)
gc_21, dtf_21, pdc_21, _    = slide_2x2(1, 0)

n_win  = gc_12.shape[1]                 # число окон
dt      = step / fs                     # 0.64 c
times   = np.arange(n_win)*dt + win_len/(2*fs)     # центры окон (как раньше)
time_edges = np.arange(n_win + 1)*dt               # ← теперь 0, 0.64, 1.28 … 10
df         = freqs[1] - freqs[0]
freq_edges = np.concatenate(([0], freqs[:-1] + df/2, [freqs[-1] + df/2]))

measures = {
    'GC' : (gc_12,  gc_21),
    'PDC': (pdc_12, pdc_21),
    'DTF': (dtf_12, dtf_21)
}

fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharey=True,
                        constrained_layout=True)

for col, (name, (m12, m21)) in enumerate(measures.items()):

    im = axs[0, col].pcolormesh(time_edges, freq_edges, m12, shading='auto', cmap='turbo')
    axs[0, col].set_title(f'{name}\n1 → 2')
    axs[1, col].pcolormesh(time_edges, freq_edges, m21, shading='auto', cmap='turbo')
    axs[1, col].set_title(f'2 → 1')

    # горизонтальные линии на частотах драйверов
    for ax in (axs[0, col], axs[1, col]):
        for f0 in (10, 30):
            ax.axhline(f0, ls='--', lw=1.1, color='white')
        ax.set_xlim(time_edges[0], time_edges[-1])
        ax.set_ylim(0, 50)
        ax.set_xlabel('t, c')

    # общая цветовая шкала для данного показателя
    cbar = fig.colorbar(im, ax=axs[:, col], shrink=0.85, pad=0.01)
    cbar.set_label('power')

# подпись оси частот только слева
for ax in axs[:, 0]:
    ax.set_ylabel('f, Гц')

plt.show()

