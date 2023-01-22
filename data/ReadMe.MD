The Folder contains the `.pickle` files that is used for the dashboard visualization. If you need any other fata files (csv, json) put them in this folder.

### Saving .pickle files using `VectorBT Pro` :
<br> The following is the code used to save the `.pickle` files of the price_data, indicators_data and the portfolio simulation data respectively.
```python
pf_simulation = vbt.Portfolio.from_signals(
    close = mtf_data['m15_close'], 
    entries = mtf_data['entries'], 
    exits = mtf_data['exits'], 
    direction = "both", ## This setting trades both long and short signals
    freq = pd.Timedelta(minutes=15), 
    init_cash = 100000
)

price_data = {"h4_data": h4_data, "m15_data" : m15_data}
vbt_indicators = {'m15_rsi': m15_rsi,'m15_price_bbands': m15_bbands, 'm15_rsi_bbands' : m15_bbands_rsi,
                  'h4_rsi': h4_rsi, 'h4_price_bbands':h4_bbands, 'h4_rsi_bbands' : h4_bbands_rsi}

entries_exits_data = {'clean_entries' : clean_entries, 'clean_exits' : clean_exits}

file_path1 = '../vbt_dashboard/data/price_data'
file_path2 = '../vbt_dashboard/data/indicators_data'
file_path3 = '../vbt_dashboard/data/entries_exits_data'
file_path4 = '../vbt_dashboard/data/pf_sim'


vbt.save(price_data, file_path1)
vbt.save(vbt_indicators, file_path2)
vbt.save(entries_exits_data, file_path3)
vbt.save(pf_simulation, file_path4)

print(type(h4_data), '||' ,type(m15_data))
print(type(h4_bbands), '||', type(h4_bbands_rsi), '||', type(h1_rsi))
print(type(m15_bbands), '||', type(m15_bbands_rsi), '||', type(m15_rsi))
```

These pickle files comprise of various class objects as can be seen from the output of the print statement in the above code block.

```
<class 'vectorbtpro.data.custom.HDFData'> || <class 'vectorbtpro.data.custom.HDFData'>
<class 'vectorbtpro.indicators.factory.talib.BBANDS'> || <class 'vectorbtpro.indicators.factory.talib.BBANDS'> || <class 'pandas.core.frame.DataFrame'>
<class 'vectorbtpro.indicators.factory.talib.BBANDS'> || <class 'vectorbtpro.indicators.factory.talib.BBANDS'> || <class 'pandas.core.frame.DataFrame'>
```

### Loading pickle files using `vbt` :

You can see this in action in the [`vbt_dashboard.py` script](https://github.com/diliprk/vectorbt_pro_dashboard/blob/main/vbt_dashboard.py)
```python
pf_simulation = vbt.Portfolio.load('data/pf_sim.pickle') ## Portfolio Simulation Results
price_data = vbt.load('data/price_data.pickle') ## OHLCV - MTF Price Data
indicators_data = vbt.load('data/indicators_data.pickle') ## Indicators Data
entries_exits_data = vbt.load('data/entries_exits_data.pickle') ## Entries & Exits Data
```
