# Deep Hedging in the Black–Scholes Model

Implementation of the deep hedging framework of Buehler et al. (2019) in a discrete-time Black–Scholes setting using PyTorch and Monte Carlo simulation.

This repository trains neural networks to learn a self-financing hedging strategy for a European call option and compares it against classical Black–Scholes delta hedging.

---

## Model Setup

We consider a risk-neutral Black–Scholes model:

- $S_0 = 1$
- $\sigma = 0.5$
- $T = 30/365$
- $N = 30$ rebalancing dates
- $r = 0$
- Strike $K = 1$

The option payoff is:

$$
g(S_T) = (S_T - K)^+
$$

Training set: 100,000 simulated paths  
Test set: 10,000 simulated paths


---

## Deep Hedging Objective

We train neural networks to minimize the empirical squared hedging error over Monte Carlo simulated paths.

The loss corresponds to the mean squared difference between the terminal payoff $g(S_T)$ and the discrete-time self-financing hedging portfolio:

$g(S_T) - p - \sum_{j=0}^{N-1} H_{t_j}(S_{t_j}) (S_{t_{j+1}} - S_{t_j})$

where:

- $p$ is the Black–Scholes price at $t=0$
- $H_{t_j}$ are neural networks representing hedge ratios
- Rebalancing occurs at discrete time steps



The learned hedge is evaluated via the out-of-sample P&L distribution.

---

## Notebook Contents

- Exact simulation of Black–Scholes paths
- Neural network training (multi-network or time-conditioned variant)
- Hedging P&L distribution analysis (mean, variance, histogram)
- Comparison with discrete-time Black–Scholes delta hedging
- Visualization of learned hedge vs analytical delta

---

## Repository Structure

