https://github.com/robertmartin8/PyPortfolioOpt - 3.8k starts
https://github.com/dcajasn/Riskfolio-Lib - 2.4k starts

TODOs:

4. Delta Hedging and Black Scholes (Delta Hedging is dynamic lol)
   For Delta Hedging need to solve following problems:

- pricing Fn to allow buy/sell, not just exercise
- Currently only allow Exercising option

5. Delta, Gamma Hedging Objectves
   Delta Hedging - correlation between Asset price and return = 0
   Gamma Hedging - correlation between Asset price and return for different bins of asset prices = 0

Improvements:

3. Move Prepending with ones into the asset object as well as cumprod (makes very hard to develop extern)
4. Negaitve weights (no)
5. Make Tests more deterministic
6. Simplify Macros/ Make more reusable
7. Option pricing/Implied Volatility - increase precision (two portfolio optimizer??, start with Variance Minimization, should be the easiest)
8. Do not rely on csv logs when doing tests
9. Maturity and Lock-In for deposits
10. Metric Class to define what fields to use right away

Today work:

2. Comission and Initial Values
3. Deep AR Multi Stock
4. Proper Liquidations
5. DeepAR integration
6. Write tests
7. Investigate permutes in intrument.py calculateProfit and stategy.py estimateProfit
