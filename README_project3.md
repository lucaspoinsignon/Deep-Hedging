# Project 3 — Deep Hedging (Black–Scholes)

Course project for **Machine Learning in Finance & Insurance (ETH Zurich, Fall 2024)**.

This repository implements the *deep hedging* approach of Buehler et al. (2019) in the Black–Scholes model, trained on simulated paths to learn a discrete-time self-financing hedging strategy for a European call option.

## Problem setup
- Risky asset follows Black–Scholes dynamics (risk-neutral, r = 0)
- Parameters: \(S_0=1\), \(\sigma=0.5\), \(T=30/365\), \(N=30\) time steps
- Option: European call payoff \(g(S_T) = (S_T - K)_+\) with \(K=1\)
- Training set: 100,000 simulated paths
- Test set: 10,000 simulated paths

## Method
We minimize the empirical deep-hedging loss on simulated paths:
\[
\frac{1}{m}\sum_{i=1}^m\Big(g(s_T^{(i)}) - p - \sum_{j=0}^{N-1} H_{t_j}(\log(s_{t_j}^{(i)}))\,(s_{t_{j+1}}^{(i)}-s_{t_j}^{(i)})\Big)^2
\]
where \(p\) is the Black–Scholes call price at \(t=0\) and \(H_{t_j}\) are neural networks (or a single time-conditioned network, depending on the exercise).

The notebook includes:
- Exact simulation of Black–Scholes paths on the grid
- Deep hedging training (multi-network or single-network variant)
- Evaluation of hedging P&L distribution on a test set (histogram + mean/std)
- Comparison with analytical Black–Scholes delta hedging (discrete rebalancing)
- Plots comparing learned hedge functions vs analytical delta for selected times

## Repo structure
```
Credit-Analytics/
├── notebooks/
│   └── deep_hedging_project3.ipynb
├── requirements.txt
└── README.md
```

## Running
Install dependencies:
```bash
pip install -r requirements.txt
```

Open the notebook and run all cells.

## Notes
- The notebook is designed to follow the official assignment steps and uses simulated (synthetic) data.
- If you use Apple Silicon / CPU-only PyTorch, ensure your local PyTorch install matches your machine (the `torch` line in requirements may need adjustment).

## Reference
Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). *Deep Hedging*. Quantitative Finance, 19(8):1271–1291.
