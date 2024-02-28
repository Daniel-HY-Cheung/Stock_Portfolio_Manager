import datetime as dt  		  	   		  	  		  		  		    	 		 		   		 		  
import random  		  	   		  	  		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
import util as ut
import indicators as IND
import numpy as np
import BagLearner as BL
import RTLearner as RTL
  		  	   		  	  		  		  		    	 		 		   		 		    		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
  	   		  	  		  		  		    	 		 		   		 		  
    # constructor  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(self, impact=0.0, commission=0.0): 	   		  	  		  		  		    	 		 		   		 		  	  	  		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  	  		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = BL.BagLearner(learner = RTL.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)

    # this method trains the learner		  	   		  	  		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
      
        # generate prices and empty trades dataframe
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        trades = prices.copy()
        trades.values[:] = 0

        # generate indicator data, convert to x array of data
        BollingerBands = IND.BollingerBands(prices,20)
        EMA = IND.EMA(prices,20)
        Price_over_EMA = prices/EMA
        CCI = IND.CCI(prices,20)
        BollingerBands = BollingerBands[40:]
        Price_over_EMA = Price_over_EMA[40:]
        CCI = CCI[40:]
        temp = pd.concat([BollingerBands,CCI,Price_over_EMA],axis=1)
        Xn = temp.to_numpy()

        # find Y data points and transform to np array
        N = 8
        Y = prices.copy()
        Y.values[:] = 0
        YBUY = 0.05 + self.impact
        YSELL = -0.05 + self.impact

        for i in range(0,len(prices)-N):
            ret = (prices.values[i+N]/prices.values[i]) - 1.0
            if ret > YBUY:
                Y.values[i] = 1
            elif ret < YSELL:
                Y.values[i] = -1
        Y = Y[40:].to_numpy()
        Y = np.transpose(Y)
        Y = Y[0]

        # construct a bunch of random trees
        self.learner.add_evidence(Xn,Y)

    # returns trade calls based on strategy		  	   		  	  		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):  		  	   		  	  		  		  		    	 		 		   		 		    	   		  	  		  		  		    	 		 		   		 		  	   		  	  		  		  		    	 		 		   		 		  
	  	   		  	  		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		  	  		  		  		    	 		 		   		 		  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
        trades = prices_all[[symbol,]]  # only portfolio symbols
        prices = prices_all[[symbol,]]
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  	  		  		  		    	 		 		   		 		  
        trades.values[:, :] = 0  # set them all to nothing

        # find indicator values
        BollingerBands = IND.BollingerBands(prices,20)
        EMA = IND.EMA(prices,20)
        Price_over_EMA = prices/EMA
        CCI = IND.CCI(prices,20)
        BollingerBands = BollingerBands[40:]
        Price_over_EMA = Price_over_EMA[40:]
        CCI = CCI[40:]
        temp = pd.concat([BollingerBands,CCI,Price_over_EMA],axis=1)
        Xn = temp.to_numpy()

        # query test points and get results from baglearner
        results = self.learner.query(Xn)
        position = 0
        for i in range(0,len(results)):
            if(results[i] == 1) and (position != 1000):
                trades.values[i+40] = 1000 - position
                position = 1000
            elif(results[i] == -1) and (position != -1000):
                trades.values[i+40] = -1000 - position
                position = -1000
        return trades  		  	   	
