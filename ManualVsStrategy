import matplotlib.pyplot as plt
import datetime as dt
import ManualStrategy as ms
import StrategyLearner as sl
import marketsimcode as marketsim

def main():
    # set constants
    sym = 'JPM'
    start_val = 100000

    # calculate Manual Strategy trades
    MSLearner = ms.ManualStrategy(verbose=False,commission=9.95,impact=0.0)
    Manual_Trades_in = MSLearner.testPolicy(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31),sv=start_val)
    Manual_Trades_in.drop(Manual_Trades_in.index[0:40], inplace=True)
    Manual_Trades_out = MSLearner.testPolicy(symbol=sym, sd=dt.datetime(2009, 11, 4), ed=dt.datetime(2011, 12, 31),sv=start_val)
    Manual_Trades_out.drop(Manual_Trades_out.index[0:40], inplace=True)

    # create benchmark trades
    benchmark_in = Manual_Trades_in.copy()
    benchmark_in.values[:] = 0
    benchmark_in.values[0] = 1000
    benchmark_out = Manual_Trades_out.copy()
    benchmark_out.values[:] = 0
    benchmark_out.values[0] = 1000

    # calculate Strategy Learner trades
    Slearner = sl.StrategyLearner(verbose=False,commission=9.95,impact=0.0)
    Slearner.add_evidence(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31), sv=start_val)
    Strategic_Trades_in = Slearner.testPolicy(symbol=sym, sd=dt.datetime(2007, 11, 2), ed=dt.datetime(2009, 12, 31),sv=start_val)
    Strategic_Trades_out = Slearner.testPolicy(symbol=sym, sd=dt.datetime(2009, 11, 4), ed=dt.datetime(2011, 12, 31),sv=start_val)

    # compute portfolio values
    Manual_Trades_portvals_in = marketsim.compute_portvals(orders_df=Manual_Trades_in, sym=sym, start_val=start_val,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),commission=9.95)
    Manual_Trades_portvals_in = Manual_Trades_portvals_in/100000
    benchmark_portvals_in = marketsim.compute_portvals(orders_df=benchmark_in, sym=sym, start_val=start_val,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),commission=9.95)
    benchmark_portvals_in = benchmark_portvals_in/100000
    Strategic_Trades_portvals_in = marketsim.compute_portvals(orders_df=Strategic_Trades_in, sym=sym,start_val=start_val, sd=dt.datetime(2008, 1, 1),ed=dt.datetime(2009, 12, 31),commission=9.95)
    Strategic_Trades_portvals_in = Strategic_Trades_portvals_in/100000

    Manual_Trades_portvals_out = marketsim.compute_portvals(orders_df=Manual_Trades_out, sym=sym,start_val=start_val, sd=dt.datetime(2010, 1, 1),ed=dt.datetime(2011, 12, 31),commission=9.95)
    Manual_Trades_portvals_out = Manual_Trades_portvals_out/100000
    Strategic_Trades_portvals_out = marketsim.compute_portvals(orders_df=Strategic_Trades_out, sym=sym,start_val=start_val, sd=dt.datetime(2010, 1, 1),ed=dt.datetime(2011, 12, 31),commission=9.95)
    Strategic_Trades_portvals_out = Strategic_Trades_portvals_out/100000
    benchmark_portvals_out = marketsim.compute_portvals(orders_df=benchmark_out, sym=sym,start_val=start_val, sd=dt.datetime(2010, 1, 1),ed=dt.datetime(2011, 12, 31),commission=9.95)
    benchmark_portvals_out = benchmark_portvals_out/100000

    # In-sample plots
    plt.clf()
    plt.plot(benchmark_portvals_in, label='Benchmark', color='purple')
    plt.plot(Manual_Trades_portvals_in, label='Manual Strategy', color='red')
    plt.plot(Strategic_Trades_portvals_in, label='Strategic Learner', color='blue')
    plt.title('In-Sample Portfolio Value')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure 4')

    # Out-of-sample plots
    plt.clf()
    plt.plot(benchmark_portvals_out, label='Benchmark', color='purple')
    plt.plot(Manual_Trades_portvals_out, label='Manual Strategy', color='red')
    plt.plot(Strategic_Trades_portvals_out, label='Strategic Learner', color='blue')
    plt.title('Out-of-Sample Portfolio Value')
    plt.xlabel('Date (YYYY-MM)')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure 5')
