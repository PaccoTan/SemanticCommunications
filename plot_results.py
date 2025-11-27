import math

import matplotlib.pyplot as plt
import json

with open('QAM_results.json', 'r') as file:
    QAM_result = json.load(file)

with open('PSK_results.json', 'r') as file:
    PSK_result = json.load(file)

colors = ["#1f77b4", "#2ca02c", "#d62728"]
markers = ["o", "s", "^"]
linestyles = ["-", "--", ":"]
for name, result in QAM_result.items():
    K,SNR = name.split('_')
    i = int(math.log2(int(K))//2-1)
    plt.plot(
        result["SNR"],result["SSIM"],
        label=f"{K}-QAM, SNR={SNR}dB",
        color=colors[i],
        linestyle=linestyles[int(SNR)//5-i],
        marker=markers[int(SNR)//5-i],
        markersize=5,
    )
plt.xlabel("Test Channel SNR")
plt.ylabel("SSIM")
plt.legend()
plt.grid()
plt.show()