import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg


def get_eigenports(returns, eigen_cnt):
    """
    :param returns: Pandas Dataframe of Asset Returns for the period
    :param eigen_cnt: Number of EigenPortfolios to return
    :return: Pandas dataframe of Asset Weights in each EigenPortfolio
    """
    # Normalize returns and obtain eigenvalues/vectors from correlation matrix of normalized returns
    norm_rets = (returns - returns.mean()) / returns.std()
    corr_mtrx = norm_rets.corr()
    eigenvalues, eigenvectors = np.linalg.eig(corr_mtrx)

    # Sort EigenVectors from largest to smallest eigenvalue and construct eigen_cnt eigenportfolios
    e_vec_df = pd.DataFrame([np.real(x) for _, x in sorted(zip(eigenvalues, eigenvectors), reverse=True)])
    e_vec_df.columns = norm_rets.columns
    e_vec_df = e_vec_df.drop(range(eigen_cnt, len(e_vec_df.index)), axis=0)

    # Avellenada divides each vector by the vector of asset volatilities,
    # though these are not orthogonal anymore.... Is that a good thing?
    # Let's Verify returns are uncorrelated and then move on

    eigen_portfolios = e_vec_df / returns.std()
    eigen_portfolios = eigen_portfolios.div(eigen_portfolios.sum(axis=1), axis=0)

    return eigen_portfolios


def get_loadings_and_s_score(returns, eigen_ports):
    """
    This function calculates eigenportfolio returns for a specified return period and then regresses the individual
    asset returns against these generated eigen-return-factors. The residuals for each stock from this regression are
    fit to an Ornstein-Uhlembeck process for generating a mean-reversion trading signal.

    :param returns: Pandas Dataframe of Asset Returns for the period
    :param eigen_ports: Pandas dataframe of Asset Weights in each EigenPortfolio
    :return: Loadings (betas) for each asset to each EigenPortfolio and the fitted OU parameters
    """

    e_port_rets = returns.dot(eigen_ports.transpose())
    # print(e_port_rets)
    # Should these returns be uncorrelated?

    resids_df = pd.DataFrame()
    betas_df = pd.DataFrame()
    for stock in returns.columns:
        regr = LinearRegression()
        regr.fit(e_port_rets, returns[stock])
        predict = pd.DataFrame(regr.predict(e_port_rets), index=e_port_rets.index, columns=[stock])
        resids_df[stock] = predict[stock] - returns[stock]
        betas_df[stock] = regr.coef_

    # Model cumulative residuals of regressed returns using eigen-factors as an Ornstein-Uhlembeck Process
    # Can be fit to an AR(1) model using analytic solution to OU SDE
    resids_df = resids_df.cumsum()
    ou_df = pd.DataFrame(index=['k', 'm', 'sigma', 'sigma_eq'])
    for stock in resids_df.columns:
        model = AutoReg(resids_df[stock], lags=1).fit()
        a = model.params['const']
        b = model.params[stock + '.L1']

        k = math.log(b) * -252
        m = a / (1 - b)
        sigma = (model.resid.var() * 2 * k / (1 - b ** 2)) ** (1/2)
        sigma_eq = (model.resid.var() / (1 - b ** 2)) ** (1/2)

        ou_df[stock] = [k, m, sigma, sigma_eq]

    ou_df.loc['m_bar'] = ou_df.loc['m'] - ou_df.mean(axis=1)['m']
    ou_df.loc['s_score'] = -ou_df.loc['m_bar'] / ou_df.loc['sigma_eq']

    return betas_df, ou_df


class PCA_Arbitrage_Strategy:
    def __init__(self, tickers, start, end, corr_lookback=252, ou_lookback=60, eigen_cnt=15, k_min=252/30,
                 long_op=-1.25, long_cl=-0.5, short_op=1.25, short_cl=0.75):
        """
        :param tickers: List of stock tickers for asset universe
        :param start: %Y-%m-%d String, first trading day (e.g. '2021-01-01)
        :param end: %Y-%m-%d String, first trading day (e.g. '2021-01-02)
        :param corr_lookback: Int, number of leading trading days of returns to use in PCA decomposition
        :param ou_lookback: Int, number of leading trading days of returns to use in OU residuals process
        :param eigen_cnt: Int, number of eigenportfolios to generate
        :param k_min: minimum mean reversion factor in OU process for stock to be considered
        :param long_op: s_score to open a long stock / short eigen trade
        :param long_cl: s_score to close a long stock / short eigen trade
        :param short_op: s_score to open a short stock / long eigen trade
        :param short_cl: s_score to close a short stock / long eigen trade
        """
        self._tickers = tickers
        self._start = datetime.strptime(start, '%Y-%m-%d')
        self._end = datetime.strptime(end, '%Y-%m-%d')
        self._corr_lookback = corr_lookback
        self._ou_lookback = ou_lookback
        self._eigen_cnt = eigen_cnt
        self._k_min = k_min
        self._long_op = long_op
        self._long_cl = long_cl
        self._short_op = short_op
        self._short_cl = short_cl

        if ou_lookback > corr_lookback:
            raise ValueError('Recommend using at least as many returns in PCA decomposition as signal generation')

        data_start = self._start - timedelta(math.ceil(corr_lookback) * 365/252)
        self._prices = yf.download(tickers, data_start, self._end)['Adj Close']
        self._returns = self._prices.pct_change().dropna()

    def simulate(self):
        short_arbs = []
        long_arbs = []
        active_arbs = pd.DataFrame(columns=self._returns.columns)
        trade_log = pd.DataFrame(columns=self._returns.columns)

        for i in range(len(self._returns.index) - self._corr_lookback - 1):
            corr_ret_sub = self._returns.iloc[i:(i+self._corr_lookback)]
            ou_ret_sub = self._returns.iloc[(self._corr_lookback - self._ou_lookback + i):(i+self._corr_lookback)]
            daily_order = pd.DataFrame(columns=self._returns.columns)

            eigen_ports = get_eigenports(corr_ret_sub, self._eigen_cnt)
            betas, ou_fitting = get_loadings_and_s_score(ou_ret_sub, eigen_ports)

            # Generate positions for new long arbitrage trades
            potential_longs = ou_fitting.loc[:,
                              (ou_fitting.loc['s_score'] < self._long_op) & (ou_fitting.loc['k'] > self._k_min)]
            for trade in potential_longs.columns:
                if trade not in long_arbs+short_arbs:
                    long_arbs.append(trade)

                    temp_df = eigen_ports.multiply(-betas[trade], axis="index")
                    temp_df.loc[self._eigen_cnt] = np.zeros(len(temp_df.columns))
                    temp_df.loc[self._eigen_cnt, trade] = 1

                    active_arbs.loc[trade] = temp_df.sum()
                    daily_order.loc[trade] = temp_df.sum()

            # Generate positions for new short arbitrage trades
            potential_shorts = ou_fitting.loc[:,
                              (ou_fitting.loc['s_score'] > self._short_op) & (ou_fitting.loc['k'] > self._k_min)]
            for trade in potential_shorts.columns:
                if trade not in long_arbs + short_arbs:
                    short_arbs.append(trade)

                    temp_df = eigen_ports.multiply(betas[trade], axis="index")
                    temp_df.loc[self._eigen_cnt] = np.zeros(len(temp_df.columns))
                    temp_df.loc[self._eigen_cnt, trade] = -1

                    active_arbs.loc[trade] = temp_df.sum()
                    daily_order.loc[trade] = temp_df.sum()

            # Check for open long/short positions that can be closed
            for trade in potential_longs.columns:
                if trade in short_arbs:
                    daily_order.loc[trade] = -active_arbs.loc[trade]
                    active_arbs = active_arbs.drop(trade, axis=0)
                    short_arbs.remove(trade)

            for trade in potential_shorts.columns:
                if trade in long_arbs:
                    daily_order.loc[trade] = -active_arbs.loc[trade]
                    active_arbs = active_arbs.drop(trade, axis=0)
                    long_arbs.remove(trade)

            trade_log.loc[self._returns.index[i+self._corr_lookback+1]] = daily_order.sum()

        return trade_log


if __name__ == '__main__':
    djia = ['UNH', 'MSFT', 'GS', 'HD', 'MCD', 'CAT', 'AMGN', 'V', 'CRM',
                'HON', 'AAPL', 'TRV', 'AXP', 'JNJ', 'CVX', 'WMT', 'PG', 'JPM', 'IBM',
                'NKE', 'MRK', 'MMM', 'DIS', 'KO', 'CSCO', 'VZ', 'INTC', 'WBA']

    # BA, DOW working?

    #prices = yf.download(tickers=djia, start=datetime(2021, 1, 1), end=datetime(2023, 1, 1))['Adj Close']
    #rets = prices.pct_change().dropna()

    #e_ports = get_eigenports(returns=rets, eigen_cnt=5)
    #print(e_ports)

    #print(get_loadings_and_s_score(rets, e_ports))

    test = PCA_Arbitrage_Strategy(djia, '2018-01-01', '2018-02-01')
    print(test.simulate())

