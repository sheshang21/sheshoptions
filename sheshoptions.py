import React, { useState, useEffect, useRef } from 'react';
import { Search, RefreshCw, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

const LiveOptionsPricing = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMarket, setSelectedMarket] = useState('NSE');
  const [optionsData, setOptionsData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);
  const refreshInterval = useRef(null);

  // Sample underlying stocks for different markets
  const marketStocks = {
    NSE: ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
    NYSE: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    NASDAQ: ['NVDA', 'META', 'NFLX', 'AMD', 'INTC']
  };

  // Black-Scholes formula
  const normalCDF = (x) => {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - prob : prob;
  };

  const calculateBlackScholes = (S, K, T, r, sigma, type) => {
    if (T <= 0) return type === 'call' ? Math.max(S - K, 0) : Math.max(K - S, 0);
    
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    
    if (type === 'call') {
      return S * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
    } else {
      return K * Math.exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
    }
  };

  const calculateGreeks = (S, K, T, r, sigma, type) => {
    if (T <= 0) return { delta: 0, gamma: 0, theta: 0, vega: 0 };
    
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    const nd1 = normalCDF(d1);
    const npd1 = Math.exp(-d1 * d1 / 2) / Math.sqrt(2 * Math.PI);
    
    const delta = type === 'call' ? nd1 : nd1 - 1;
    const gamma = npd1 / (S * sigma * Math.sqrt(T));
    const theta = type === 'call' 
      ? (-S * npd1 * sigma / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normalCDF(d2)) / 365
      : (-S * npd1 * sigma / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normalCDF(-d2)) / 365;
    const vega = S * npd1 * Math.sqrt(T) / 100;
    
    return { delta, gamma, theta, vega };
  };

  const fetchStockPrice = async (symbol) => {
    try {
      const response = await fetch(
        `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1m&range=1d`
      );
      const data = await response.json();
      
      if (data.chart.result && data.chart.result[0]) {
        const quote = data.chart.result[0].meta;
        return {
          price: quote.regularMarketPrice,
          change: quote.regularMarketPrice - quote.previousClose,
          changePercent: ((quote.regularMarketPrice - quote.previousClose) / quote.previousClose) * 100
        };
      }
      return null;
    } catch (err) {
      console.error(`Error fetching ${symbol}:`, err);
      return null;
    }
  };

  const generateOptions = (stockSymbol, stockPrice) => {
    const strikes = [
      Math.round(stockPrice * 0.85),
      Math.round(stockPrice * 0.90),
      Math.round(stockPrice * 0.95),
      Math.round(stockPrice),
      Math.round(stockPrice * 1.05),
      Math.round(stockPrice * 1.10),
      Math.round(stockPrice * 1.15)
    ];
    
    const expirations = [
      { date: '2025-12-30', label: '30 Dec 25' },
      { date: '2026-01-30', label: '30 Jan 26' },
      { date: '2026-03-27', label: '27 Mar 26' }
    ];
    
    const options = [];
    const r = 0.065; // Risk-free rate ~6.5%
    const sigma = 0.30; // Implied volatility ~30%
    
    expirations.forEach(exp => {
      const T = (new Date(exp.date) - new Date()) / (1000 * 60 * 60 * 24 * 365);
      
      strikes.forEach(strike => {
        ['call', 'put'].forEach(type => {
          const theoreticalPrice = calculateBlackScholes(stockPrice, strike, T, r, sigma, type);
          const intrinsicValue = type === 'call' 
            ? Math.max(stockPrice - strike, 0) 
            : Math.max(strike - stockPrice, 0);
          const timeValue = theoreticalPrice - intrinsicValue;
          const greeks = calculateGreeks(stockPrice, strike, T, r, sigma, type);
          
          const optionCode = `${stockSymbol.replace('.NS', '')} ${exp.label} ${strike} ${type === 'call' ? 'CE' : 'PE'}`;
          
          options.push({
            code: optionCode,
            underlying: stockSymbol,
            strike,
            type,
            expiry: exp.label,
            expiryDate: exp.date,
            spotPrice: stockPrice,
            theoreticalPrice,
            intrinsicValue,
            timeValue,
            ...greeks,
            itm: type === 'call' ? stockPrice > strike : stockPrice < strike
          });
        });
      });
    });
    
    return options;
  };

  const fetchAllOptions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const stocks = marketStocks[selectedMarket];
      const allOptions = [];
      
      for (const stock of stocks) {
        const stockData = await fetchStockPrice(stock);
        if (stockData) {
          const options = generateOptions(stock, stockData.price);
          allOptions.push(...options);
        }
      }
      
      setOptionsData(allOptions);
      setLastUpdate(new Date());
    } catch (err) {
      setError('Failed to fetch options data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllOptions();
    
    // Auto-refresh every 30 seconds
    refreshInterval.current = setInterval(() => {
      fetchAllOptions();
    }, 30000);
    
    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
      }
    };
  }, [selectedMarket]);

  const filteredOptions = optionsData.filter(opt => 
    opt.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
    opt.underlying.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Live Options Pricing Model
          </h1>
          <p className="text-slate-400">Real-time options data with Black-Scholes pricing and Greeks</p>
        </div>

        {/* Controls */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 mb-6 border border-slate-700">
          <div className="flex flex-wrap gap-4 items-center">
            {/* Search */}
            <div className="flex-1 min-w-[300px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
                <input
                  type="text"
                  placeholder="Search by option code (e.g., RELIANCE 30 Dec 25 1550 CE)"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-slate-900/50 border border-slate-600 rounded-lg focus:outline-none focus:border-blue-500 text-white placeholder-slate-500"
                />
              </div>
            </div>

            {/* Market Selection */}
            <select
              value={selectedMarket}
              onChange={(e) => setSelectedMarket(e.target.value)}
              className="px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-lg focus:outline-none focus:border-blue-500 text-white"
            >
              <option value="NSE">NSE (India)</option>
              <option value="NYSE">NYSE (US)</option>
              <option value="NASDAQ">NASDAQ (US)</option>
            </select>

            {/* Refresh Button */}
            <button
              onClick={fetchAllOptions}
              disabled={loading}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <RefreshCw size={20} className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>

          {/* Status Bar */}
          <div className="mt-4 flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <span className="text-slate-400">
                {filteredOptions.length} options displayed
              </span>
              {lastUpdate && (
                <span className="text-slate-500">
                  Last update: {lastUpdate.toLocaleTimeString()}
                </span>
              )}
            </div>
            {loading && (
              <span className="text-blue-400 flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                Updating...
              </span>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-700 rounded-lg flex items-center gap-2 text-red-400">
              <AlertCircle size={20} />
              {error}
            </div>
          )}
        </div>

        {/* Options Table */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-900/50 border-b border-slate-700">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Option Code</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Spot</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Strike</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Type</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Theoretical</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Intrinsic</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Time Value</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Delta</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Gamma</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Theta</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-300 uppercase tracking-wider">Vega</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700">
                {filteredOptions.map((option, idx) => (
                  <tr 
                    key={idx} 
                    className={`hover:bg-slate-700/30 transition-colors ${
                      option.itm ? 'bg-green-900/10' : ''
                    }`}
                  >
                    <td className="px-4 py-3 text-sm font-medium">
                      <div className="flex items-center gap-2">
                        {option.code}
                        {option.itm && (
                          <span className="px-2 py-0.5 bg-green-600/20 text-green-400 text-xs rounded">ITM</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <div className="flex items-center gap-1">
                        ₹{option.spotPrice.toFixed(2)}
                        {option.spotPrice > option.strike ? (
                          <TrendingUp size={14} className="text-green-400" />
                        ) : (
                          <TrendingDown size={14} className="text-red-400" />
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">₹{option.strike}</td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        option.type === 'call' 
                          ? 'bg-blue-600/20 text-blue-400' 
                          : 'bg-purple-600/20 text-purple-400'
                      }`}>
                        {option.type === 'call' ? 'CALL' : 'PUT'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-right font-medium text-blue-400">
                      ₹{option.theoreticalPrice.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right font-medium text-green-400">
                      ₹{option.intrinsicValue.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-slate-300">
                      ₹{option.timeValue.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-slate-300">
                      {option.delta.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-slate-300">
                      {option.gamma.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-slate-300">
                      {option.theta.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-slate-300">
                      {option.vega.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredOptions.length === 0 && !loading && (
            <div className="p-12 text-center text-slate-400">
              <AlertCircle className="mx-auto mb-4" size={48} />
              <p className="text-lg">No options found matching your search</p>
              <p className="text-sm mt-2">Try adjusting your search query or market selection</p>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="mt-6 bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-slate-700">
          <h3 className="text-sm font-semibold mb-3 text-slate-300">Understanding the Greeks:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs text-slate-400">
            <div>
              <span className="font-medium text-slate-300">Delta:</span> Rate of change of option price vs underlying
            </div>
            <div>
              <span className="font-medium text-slate-300">Gamma:</span> Rate of change of delta
            </div>
            <div>
              <span className="font-medium text-slate-300">Theta:</span> Time decay per day
            </div>
            <div>
              <span className="font-medium text-slate-300">Vega:</span> Sensitivity to volatility (per 1% change)
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-3">
            Note: Theoretical prices calculated using Black-Scholes model with 6.5% risk-free rate and 30% implied volatility
          </p>
        </div>
      </div>
    </div>
  );
};

export default LiveOptionsPricing;
