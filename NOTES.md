https://github.com/robertmartin8/PyPortfolioOpt - 3.8k starts
https://github.com/dcajasn/Riskfolio-Lib - 2.4k starts

TODOs:

4. Delta Hedging and Black Scholes (Delta Hedging is dynamic lol)
   For Delta Hedging need to solve following problems:

- pricing Fn to allow buy/sell, not just exercise
- Currently only allow Exercising option
- Delta Hedging - correlation between Asset price and return = 0
- Gamma Hedging - correlation between Asset price and return for different bins of asset prices = 0

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
3. Proper Liquidations
4. Write tests (including DeepAR)
5. Investigate permutes in intrument.py calculateProfit and stategy.py estimateProfit

2) Add macro for deepvar example
3) Add demo file
4) Add links to code
5) Remove unnecessary files
6) Add requirements.txt
