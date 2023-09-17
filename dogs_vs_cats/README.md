# dogs_vs_cats

```resnet_train.py``` : とりあえず写経  
```dog_vs_cats.ipynb``` : 実験的にコードを実行したい

## 実行
### resnet_train.py
```python
python resnet_train.py
```
|options|description|
|:--:|:--:|
|`--data_dir`|dogs-vs-catsの画像データが格納されているディレクトリ（指定必須）
|`--out_dir`|推論結果を格納するディレクトリ|
|`--forecasts`|このオプションをつけることで推論も実行|
|`--device`|デバイス環境（デフォルト : cuda）|
|`--dryrun`|このオプションをつけることでdryrun（1件のデータ）で実行|

## reference
https://www.amazon.co.jp/dp/4065305136?psc=1&ref=ppx_yo2ov_dt_b_product_details
