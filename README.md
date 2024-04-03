# Torch-Finance (ToFina)

Differential modelling for financial engineering, including portfolio optimization and derivative pricing.

The goal is to create a toolkit that goes beyond common assumption in option pricing literature, particularly risk neutrality, assumption on probability distributions of assets and no transaction costs.

Tofina models an investor who has certain risk preferences and has choice to invest into multiple instruments that track performance of multiple assets. The option pricing is from the point of view of an investor: i.e. what is the minimal price that makes the investor want to buy the derivative.

Tofina treats everything as a differentiable parameter, be it a portfolio weight, probability distribution parameter, comissions on an instrument or interest rate. Asset processes are simulated using Monte Carlo simulation, portfolio allocations are optimized to provide good average utility over all possible scenarios.

Depending on your goal, you can freeze certain parameters and optimize the rest. If you want to find implied volatility, you freeze prices and optimize the standard deviation parameter. If you want to find optimial portfolio allocation, you freeze everything but portfolio weights and try to maximize utility. If you want to find an implied interest rate or even market expectations, you can do that too.

The project is inspired after reading first four chapters of Tomas Bjork "Arbitrage Theory in Continuous Time".
Since university I was unhappy with assumption made in finance theory. After working for some years with deep learning and PyTorch, I can finally propose an alternative

# Target Audience

The project aims to be useful to finance researchers and trading practitioners that have derivatives in their potfolio. Traders could benefit from pricing models and portfolio optimization models that explicitly account for their risk preferances and any other priors. Researchers can benefit from this framework when modelling market expectations.

# Illustrative Mathematics

Classes in the code try to emulate the following equations:

$A(s, t) = D(s, t, P_A)$ - Asset process, relies on reparametrisation trick to make it differentiable with respect to $P_A$

$I(s,t) = C(A(s, t), P_I)$ - Instrument (derivative) is defined by a contract (payout) function C applied to generated asset values. The contract function includes things like comission too

$\Pi(s,t) = \Delta w(s,t, P_W) \* I(s,t)$ - Profit is defined as liquidation times holding of certain instrument

$U = \sum_t \sum_s{u(\Pi(s,t), t, P_U)}$ - Investors utility is sum over utility in each period and each scenario. Money utility function $u$ determines if the investor is risk loving or risk averse

$s$ is a Monte-Carlo scenario and $t$ is time period

$P_A, P_I, P_W, P_U$ - are all differentiable parameters that can be optimized for

Portfolio Optimisation: $w = argmax_w U$

Implied Volatility: $\sigma = argmin_{\sigma} (U - U')$ where $U'$ is the utility of portfolio without the derivative

Derivative Pricing: $P = argmin_{P} (U - U)'$ where $U'$ is the utility of portfolio without the derivative

# What can Tofina already do and what is the vision?

- [x] Replicated theoritical results for portfolio allocations in Binomial [[Code](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/tests/integration/theoryVerification/test_binomial.py#L48)] and Variance Minimization [[Code](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/tests/integration/theoryVerification/test_varianceMinimization.py)] models
- [x] Replicated theoretical results for the Black-Scholes [[Code](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/tests/integration/theoryVerification/test_BlackScholes.py#L48)] and Binomial [[Code](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/tests/integration/theoryVerification/test_binomial.py#L141)] option pricing
- [x] Replicated theoretical results for Black-Scholes implied volatility [[Code](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/tests/integration/theoryVerification/test_BlackScholes.py#L91)] (unstable at the moment)
- [x] Simple single asset trading system on real data using ARCH package to represent asset process [[Code](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/tests/integration/backtest/test_univariateGARCHstock.py)]
- [x] Simple multiasset trading system on real data using DeepVAR model from pytorch-forecasting to represent asset process [[demo.ipynb](https://github.com/MangoDataIO/tofina/blob/main/demo.ipynb)]
- [ ] Improve implied volatility and other parameter estimation
- [ ] More sophisticated entry/exit strategy than Buy-And-Hold (use NN to sell/exercise options before expiry date)
- [ ] Fitting and evaluation of different generative processes on real data
- [ ] GPU support
- [ ] Model zoo for time-series generative processes
- [ ] Set up proper backtesting for generated strategies
- [ ] Integration with data providers and trading platforms

# How can I start using Tofina?

Check the links to code in the section above. [requirements.txt](https://github.com/MangoDataIO/tofina/blob/1e0cfcd8118ffef6f9ae20d374e2fe46043e3561/requirements.txt) file with all the dependancies is included
