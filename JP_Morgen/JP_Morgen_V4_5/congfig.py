# config.py
import os

LOG_ROOT = "./log"
SAVE_DIR = './v3_data'

STOCK_UNIVERSE = {
    'mega_tech': ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','ORCL','CRM','ADBE','AMD','INTC','CSCO','IBM'],
    'semicon': ['QCOM','TXN','MU','AMAT','LRCX','KLAC','MRVL','ON','ADI','SNPS','CDNS','NXPI','MCHP','SWKS','TER'],
    'cloud_saas': ['NOW','PANW','CRWD','NET','DDOG','ZS','SNOW','PLTR','FTNT','ANET','INTU','WDAY','TEAM','HUBS','VEEV','OKTA','MDB','BILL','ZI','ESTC'],
    'consumer_internet': ['NFLX','PYPL','SQ','SHOP','UBER','ABNB','SNAP','PINS','DASH','HOOD','APP','RBLX'],
    'energy_oil': ['XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','DVN','HES','HAL','BKR','FANG','KMI'],
    'energy_clean': ['FSLR','NEE','AES','CEG','VST','NRG','GEV','ENPH','DUK','SO'],
    'financials': ['JPM','GS','MS','BAC','WFC','C','SCHW','USB','PNC','BK','COF','AXP','BLK','BX','KKR','APO','ICE','CME','MCO','SPGI'],
    'insurance': ['CB','PGR','TRV','AON','ALL','MET'],
    'pharma_biotech': ['LLY','JNJ','UNH','ABBV','MRK','PFE','AMGN','GILD','BMY','REGN','VRTX','BIIB','MRNA','ILMN','SGEN','ALNY','INCY','BGNE','EXAS','PCVX'],
    'health_devices': ['ABT','TMO','ISRG','SYK','BSX','MDT','DHR','HCA','EW','DXCM','PODD','HOLX'],
    'emerging_biotech': ['CRSP','NTLA','BEAM','EDIT','RARE','TWST','RXRX','SDGR','TXG','NUVB'],
    'industrials': ['CAT','GE','RTX','HON','BA','LMT','GD','NOC','DE','ETN','PH','TT','LHX','TDG','HWM'],
    'consumer_staples': ['PG','KO','PEP','COST','WMT','MCD','SBUX','CL','MDLZ','MNST','PM','MO'],
    'consumer_disc': ['HD','LOW','TJX','ROST','NKE','MAR','HLT','RCL','GM','ORLY','BKNG','DIS'],
    'telecom': ['TMUS','VZ','T','CMCSA','WBD','CHTR'],
    'reits': ['PLD','AMT','EQIX','DLR','SPG','WELL','CCI','PSA'],
    'materials': ['LIN','SHW','ECL','APD','NEM','FCX'],
    'transport': ['UNP','CSX','NSC','FDX','UPS','WM'],
    'emerging_ai': ['PATH','AI','IONQ','SMCI','SOUN','BBAI','GFAI','AMBA'],
}

STOCK_UNIVERSE_FLAT = []
STOCK_SECTOR_MAP = {}
for sector, tickers in STOCK_UNIVERSE.items():
    for t in tickers:
        if t not in STOCK_UNIVERSE_FLAT:
            STOCK_UNIVERSE_FLAT.append(t)
            STOCK_SECTOR_MAP[t] = sector

MARKET_TICKERS = {'SPY':'SP500','QQQ':'Nasdaq100','XLE':'EnergySector','TLT':'Bond20Y','GLD':'Gold'}
START_DATE = '2018-01-01'
END_DATE   = '2025-12-31'

# 数据预处理常量
LOOKBACK = 20
SCALER_WINDOW = 80
MIN_HISTORY_YEARS = 4
MIN_STOCKS_RATIO = 0.80
MIN_STOCKS_PER_DAY = 20