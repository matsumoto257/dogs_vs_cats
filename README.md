# dogs_vs_cats

## installation
```
pip install -r requirements.txt
```

## run
```python
python main.py
```
|options|description|
|:--:|:--|
|`--data_dir`|dogs-vs-catsの画像データが格納されているディレクトリ（指定必須）|
|`--config_path`|config.yamlのpath|
|`--out_dir`|推論結果を格納するディレクトリ|
|`--forecasts`|このオプションをつけることで推論も実行|
|`--device`|デバイス環境（デフォルト : cuda）|
|`--dryrun`|このオプションをつけることでdryrun（10件のデータ）で実行|
|`--n_epochs`|学習のエポック数（デフォルト:1）|
|`--architecture`|アーキテクチャ（指定必須）|
|`--optimizer`|optimizer（指定必須）|
|`--lr_scheduler`|学習率のscheduler|


## reference
https://www.amazon.co.jp/dp/4065305136?psc=1&ref=ppx_yo2ov_dt_b_product_details
