import os
import glob
def clear_logs():
    log_files = glob.glob("logs/*")
    for f in log_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Неуспешно изтриване на {f}: {e}")

clear_logs()
from lumibot.brokers.alpaca import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from alpaca_trade_api import REST
load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = os.getenv('BASE_URL')

ALPACA_CREDS = {
    'API_KEY': API_KEY,
    'API_SECRET': API_SECRET,
    'PAPER':True
}


class MLTrader(Strategy):
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        headlines = []
        for ev in news:
            print(ev.__dict__)
            if 'headline' in ev.__dict__:
                headlines.append(ev.__dict__['headline'])
            elif 'raw' in ev.__dict__ and 'headline' in ev.__dict__['raw']:
                headlines.append(ev.__dict__['raw']['headline'])
            else:
                headlines.append(str(ev))
        return headlines

    def initialize(self,symbol:str = 'SPY',cash_at_risk:float = .5):
        self.symbol = symbol
        self.sleeptime = '24H'
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)


    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash*self.cash_at_risk/last_price)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
    
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        headlines = []
        for ev in news:
            # Debug: виж структурата на обекта
            print(ev.__dict__)
            if 'headline' in ev.__dict__:
                headlines.append(ev.__dict__['headline'])
            elif 'raw' in ev.__dict__ and 'headline' in ev.__dict__['raw']:
                headlines.append(ev.__dict__['raw']['headline'])
            else:
                headlines.append(str(ev))
        return headlines

   

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()

        if cash > last_price:
            if self.last_trade == None:
                news = self.get_news()
                print(news )
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'buy',
                    type = 'bracket',
                    take_profit_price= last_price*1.2,
                    stop_loss_price = last_price*.95
                )
                self.submit_order(order)
                self.last_trade = 'buy'



start_date = datetime(2023,12,15)
end_date = datetime(2023,12,31)

broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(name = 'mlstrat',broker = broker,parameters={'symbol':'SPY',
                                                                 'cash_at_risk':.5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters ={'symbol':'SPY',
                 'cash_at_risk':.5},
    show_plot=True
)

