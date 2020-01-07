# deep-learning
Stock Prices Prediction Using Deep Learning Techniques 
目的是在於研究預測股票市場，股票價格是非線性時且複雜的容易遭到外在因素所影響，準確地說想要預測價格是深入學習時間序列中非常具有挑戰性的，現有的類神經網路方法中並無正確且固定的方法，同時在文字轉換、圖形辨識、語音卻能達到準確度極高的結果。

在本文的深入學習成果可應用於相關時間序列預測，介紹了使用深入學習對於股票的預測並使用yahoo提供的股票價格數據評估其表現，文內目的是比較LSTM與FACEBOOK提供的Prophet方法，使用RMSE評估精準度並繪製出圖表，對於購買者給予相關圖表進行參考，文內不針對雜訊做出排除，一併將雜訊列入原始資料中以避免數據過於合理化，將測試成果趨近於現實應用並調整三者模型參數盡可能將標準一致性已達到可驗證。

文內結果顯示Prophet儘管在預測速度上高於LSTM、LSTM-GRU、Stateful LSTM，但由於簡化了許多類神經的流程，也新增了諸多限制，對於數據分割上其為重要導致在RMSE數值較高，但在預測速度上卻遠遠高於LSTM、LSTM-GRU以及該技術提供諸多預測方式降低深入學習的門檻，本研究指出若需要在短時間內獲得預測資訊以及圖形顯示在不增加現有設備成本下該技術確實在執行時間上優於其於兩者。
關鍵字: LSTM、LSTM-GRU、Prophet、RMSE、深入學習


電腦設配簡介:
CPU:intel E3-1225v6 @ 3.3GHZ
GPU:NVIDIA Quadro P600
RAM:16G
測試設備軟體簡介:
軟體簡介:
Visual studio 2019
keras 2.2.4
python 3.6.8
Anaconda 2019.03
quandl 3.4.5
matplotlib 3.0.3
tensorflow 1.13.1
numpy 1.16.3
pystan 2.18
pandas 0.24.2
pytrends 4.4.0
fbprophet 0.4.post2


1.step  download finance.yahoo.com 所提供之歷史數據，並將數據做前處理，留下日期與[adjprice]調整後價格，
2.creat dataset 並將資料分文67%訓練 33%測試用
