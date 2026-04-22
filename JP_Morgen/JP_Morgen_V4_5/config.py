import os


MODEL_VERSION = "V4_5"
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
LOG_ROOT = os.path.join(PROJECT_DIR, "log", MODEL_VERSION)
SAVE_DIR = os.path.join(PROJECT_DIR, "v3_data")
META_FILE = os.path.join(SAVE_DIR, "meta.json")


STOCK_UNIVERSE = {
    "mega_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO", "IBM"],
    "semicon": ["QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "ADI", "SNPS", "CDNS", "NXPI", "MCHP", "SWKS", "TER"],
    "cloud_saas": ["NOW", "PANW", "CRWD", "NET", "DDOG", "ZS", "SNOW", "PLTR", "FTNT", "ANET", "INTU", "WDAY", "TEAM", "HUBS", "VEEV", "OKTA", "MDB", "BILL", "ZI", "ESTC"],
    "consumer_internet": ["NFLX", "PYPL", "SQ", "SHOP", "UBER", "ABNB", "SNAP", "PINS", "DASH", "HOOD", "APP", "RBLX"],
    "energy_oil": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "DVN", "HES", "HAL", "BKR", "FANG", "KMI"],
    "energy_clean": ["FSLR", "NEE", "AES", "CEG", "VST", "NRG", "GEV", "ENPH", "DUK", "SO"],
    "financials": ["JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "USB", "PNC", "BK", "COF", "AXP", "BLK", "BX", "KKR", "APO", "ICE", "CME", "MCO", "SPGI"],
    "insurance": ["CB", "PGR", "TRV", "AON", "ALL", "MET"],
    "pharma_biotech": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "PFE", "AMGN", "GILD", "BMY", "REGN", "VRTX", "BIIB", "MRNA", "ILMN", "SGEN", "ALNY", "INCY", "BGNE", "EXAS", "PCVX"],
    "health_devices": ["ABT", "TMO", "ISRG", "SYK", "BSX", "MDT", "DHR", "HCA", "EW", "DXCM", "PODD", "HOLX"],
    "emerging_biotech": ["CRSP", "NTLA", "BEAM", "EDIT", "RARE", "TWST", "RXRX", "SDGR", "TXG", "NUVB"],
    "industrials": ["CAT", "GE", "RTX", "HON", "BA", "LMT", "GD", "NOC", "DE", "ETN", "PH", "TT", "LHX", "TDG", "HWM"],
    "consumer_staples": ["PG", "KO", "PEP", "COST", "WMT", "MCD", "SBUX", "CL", "MDLZ", "MNST", "PM", "MO"],
    "consumer_disc": ["HD", "LOW", "TJX", "ROST", "NKE", "MAR", "HLT", "RCL", "GM", "ORLY", "BKNG", "DIS"],
    "telecom": ["TMUS", "VZ", "T", "CMCSA", "WBD", "CHTR"],
    "reits": ["PLD", "AMT", "EQIX", "DLR", "SPG", "WELL", "CCI", "PSA"],
    "materials": ["LIN", "SHW", "ECL", "APD", "NEM", "FCX"],
    "transport": ["UNP", "CSX", "NSC", "FDX", "UPS", "WM"],
    "emerging_ai": ["PATH", "AI", "IONQ", "SMCI", "SOUN", "BBAI", "GFAI", "AMBA"],
}

STOCK_UNIVERSE_FLAT = []
STOCK_SECTOR_MAP = {}
for sector, tickers in STOCK_UNIVERSE.items():
    for ticker in tickers:
        if ticker not in STOCK_UNIVERSE_FLAT:
            STOCK_UNIVERSE_FLAT.append(ticker)
            STOCK_SECTOR_MAP[ticker] = sector

MARKET_TICKERS = {"SPY": "SP500", "QQQ": "Nasdaq100", "XLE": "EnergySector", "TLT": "Bond20Y", "GLD": "Gold"}
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

LOOKBACK = 20
SCALER_WINDOW = 80
MIN_HISTORY_YEARS = 4
MIN_STOCKS_RATIO = 0.80
MIN_STOCKS_PER_DAY = 20

DEFAULT_D_MODEL = 128
DEFAULT_NUM_LAYERS = 3
DEFAULT_NHEAD = 4
DEFAULT_DROPOUT = 0.15
DEFAULT_LISTNET_WEIGHT = 0.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NUM_EPOCHS = 60
DEFAULT_PATIENCE = 8
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_DAYS = 20
DEFAULT_TOP_N = 3
DEFAULT_REBAL_FREQ = 5
DEFAULT_TX_COST = 0.001
DEFAULT_AMP_MODE = "on"
DEFAULT_AMP_DTYPE = "float16"
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_EARLY_STOP_MIN_DELTA = 0.0
