# Forex-LSTM

# Demo
## forex dataset
Inquiry Historical Foreign Exchange Rate from Bank of Taiwan <br />
https://rate.bot.com.tw/xrt/history?Lang=en-US <br />
## forex real
`$python3 readdata.py` <br />
![](./assets/2018_forex_real.jpg)

## forex predict 1 day
`$python3 rnn.py` <br />
![](./assets/201901_forex_predict.jpg)

## forex predict 5 days
`$python3 lstm.py` (train & save model then predict)<br />
![](./assets/2019_forex_predict_5days.jpg)
`$python3 lstm_load.py` (load model then predict)<br />
![](./assets/2019_forex_load_predict_5days.jpg)
