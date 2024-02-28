import matplotlib.pyplot as plt
import pandas as pd
from util import get_data
import datetime as dt
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as marketsim

def main():
    # set constants
    sym = 'JPM'
    start_val = 100000

    # calculate Strategy Learner trades
    Slearner1 = sl.StrategyLearner(verbose=False,commission=0.0,impact=0.0)
    Slearner1.add_evidence(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31), sv=start_val)
    Strategic_Trades_1 = Slearner1.testPolicy(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31),sv=start_val)

    Slearner2 = sl.StrategyLearner(verbose=False, commission=0.0, impact=0.025)
    Slearner2.add_evidence(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31), sv=start_val)
    Strategic_Trades_2 = Slearner2.testPolicy(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31),sv=start_val)

    Slearner3 = sl.StrategyLearner(verbose=False, commission=0, impact=0.05)
    Slearner3.add_evidence(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31), sv=start_val)
    Strategic_Trades_3 = Slearner3.testPolicy(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31),sv=start_val)

    # compute portfolio values
    Strategic_Trades_portvals_1 = marketsim.compute_portvals(orders_df=Strategic_Trades_1, sym=sym,start_val=start_val, sd=dt.datetime(2008, 1, 1),ed=dt.datetime(2009, 12, 31),impact=0.0)
    Strategic_Trades_portvals_1 = Strategic_Trades_portvals_1/100000
    Strategic_Trades_portvals_2 = marketsim.compute_portvals(orders_df=Strategic_Trades_2, sym=sym, start_val=start_val,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),impact=0.025)
    Strategic_Trades_portvals_2 = Strategic_Trades_portvals_2 / 100000
    Strategic_Trades_portvals_3 = marketsim.compute_portvals(orders_df=Strategic_Trades_3, sym=sym, start_val=start_val,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),impact=0.05)
    Strategic_Trades_portvals_3 = Strategic_Trades_portvals_3 / 100000

    # cumulative return calculations
    cum_ret_1 = Strategic_Trades_portvals_1.values[-1]
    cum_ret_2 = Strategic_Trades_portvals_2.values[-1]
    cum_ret_3 = Strategic_Trades_portvals_3.values[-1]

    # plot
    plt.clf()
    plt.plot(Strategic_Trades_portvals_1, label='Impact = 0.00', color='purple')
    plt.plot(Strategic_Trades_portvals_2, label='Impact = 0.025', color='red')
    plt.plot(Strategic_Trades_portvals_3, label='Impact = 0.05', color='blue')
    plt.title('Strategy Learner Portfolio Values by Impact')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.legend()
    plt.savefig('Figure 6')
