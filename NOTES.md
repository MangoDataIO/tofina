https://github.com/robertmartin8/PyPortfolioOpt - 3.8k starts
https://github.com/dcajasn/Riskfolio-Lib - 2.4k starts

2 inputs -
Utility of money U(M,t),
asset process Xt= F(It-1)

European option:
Montecarlo Xt, avg U(M,t)

Portfolio without rebalancing:
max b U(M,t) over all montecarlo draws using gradient descent

Budge constraint, self-finance portfolio?

Current Tasks:

3. Calculate Implied Volatiltiy from Real prices
4. Delta Hedging and Black Scholes (Delta Hedging is dynamic lol)
   For Delta Hedging need to solve following problems:

- pricing Fn to allow buy/sell, not just exercise
- LiquidationFN to be (-1,1) to indicate buy/sell decisions (ignore)
  (Instead of LiqduiationFN use Weighting FN)
- Currently only allow Exercising option

TODOs:

5. Delta, Gamma Hedging Objectves
   Delta Hedging - correlation between Asset price and return = 0
   Gamma Hedging - correlation between Asset price and return for different bins of asset prices = 0
6. Calibrating Asset Process and Preferences to observed option prices

Improvements:

4. Negaitve weights (no)
5. Make Tests more deterministic
6. Simplify Macros/ Make more reusable
7. Option pricing - increase precision
8. Do not rely on csv logs when doing tests
9. Maturity and Lock-In for deposits
10. REWRITE USING GLOBAL CACHE FOR COMPUTATIONS

Design Suggestion:

1. Metric Class to define what fields to use right away
2. Fix Seed in Asset to reduce variance in optimization
3. Remove Softmax in strategy normalized weights

Today work:

3. Proper Liquidations
4. Backtest reporting
5. DeepAR integration
6. Write tests
7. Investigate permutes in intrument.py calculateProfit and stategy.py estimateProfit
