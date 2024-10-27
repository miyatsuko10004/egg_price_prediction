
- Pythonファイルの実行環境準備（初回のみ）　

　windowsPCでPowerShellを実行し、以下のコマンドを実行する。

```
wsl --install
```
```
wsl.exe --install
```
PCを再起動する。

```
wsl.exe --set-default-version 2
```
ubuntuを起動
```
wsl
```

```
sudo apt update
```

```
sudo apt install python3-pip
```

python3 -m venv ~/mypy
source ~/mypy/bin/activate


pandasをインストール
```
pip3 install pandas
``` 

statsmodelsをインストール
```
pip3 install statsmodels
```

sklearnをインストール
```
pip3 install scikit-learn
```

openpyxlをインストール
```
pip3 install openpyxl
```

インストール内容を保存しておく
```
pip3 freeze > python.txt
```



- Pythonファイルの実行環境準備（2回目以降）　

windowsPCでPowerShellを実行し、以下のコマンドを実行する。

ubuntuを起動
```
wsl
```
処理のディレクトリに移動
```
cd egg_price_prediction
```
実行
```
./run.sh
```

