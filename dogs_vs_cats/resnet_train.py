import argparse
import pathlib
import numpy as np
import sklearn.model_selection
from torch import utils
import torch
import torchvision
import torchvision.transforms.functional
from torchvision import transforms
import os



# trainデータをtrain_val_splitし、trainのインデックスとvalのインデックをそれぞれ取得
def setup_train_val_split(labels, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    #sklearn.model_selection.StratifiedShuffleSplit
    #データセット全体のクラスの偏りを保持しながら、データセットを分割.データの重複が許容されていると分かる.必ずしも全てのデータがvalidationのデータセットに一度使われるわけではない.
    #検証用データが重複したり，ある学習用データが学習されなかったりするので，あまり使われないイメージ
    #インスタンスの作成
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, 
        train_size=0.8, 
        random_state=seed    #n_splits:分割&シャッフル回数
    )
    #上で分割したtrainデータのインデックス=train_indices,valデータのインデックス=val_indices
    #x:分割するデータのインデックスの配列,y:それに対応するラベルの配列
    #next()でsplitter.split(x, y)から要素を取り出す
    train_indices, val_indices = next(splitter.split(x, y))
    #dryrun=True→ランダムに100個run
    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace=False)
        val_indices = np.random.choice(val_indices, 100, replace=False)

    return train_indices, val_indices

# データセットの作成（ImageFolder）
def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

# trainデータの正解ラベル(dog=1,cat=0)を返す
def get_labels(dataset):
    # 結局このif文は何がしたいのか分からない、このif文はfalseになる
    if isinstance(dataset, torch.utils.data.Subset):
        return get_labels(dataset.dataset)[dataset.indices]
    else:
        #traiデータの正解ラベルを全て１次元配列で出力
        return np.array([img[1] for img in dataset.imgs])  # torchvision.datasets.ImageFolder.imgs : List of (image path, class_index) tuples

#setup_train_val_splitで分割したデータをdatasetに変換
def setup_train_val_datasets(data_dir, dryrun=False):
    #torchvision.datasets.ImageFolderを使用してデータセットの作成
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),   #trainデータの入っているディレクトリのパス
        transform=setup_center_crop_transform()   #transformを指定することによって画像に対する前処理をすることができる
    )
    # labels=正解ラベルの配列
    labels = get_labels(dataset)
    #etup_train_val_split
    train_indices, val_indices = setup_train_val_split(labels, dryrun)

    #訓練データセットと検証データセットにデータセットを分割する際はSubsetクラスを用いる
    #元のデータセットから指定したインデックスだけ使用するデータセットを作成できる.学習及びテストに使用するインデックスを予め作成しておくことで、データセットを分割できる.
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    return train_dataset, val_dataset

def setup_train_val_loaders(data_dir, batch_size, dryrun=False):
    #setup_train_val_datasetsでデータセットを分割
    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun=dryrun
    )
    #torch.utils.data.DataLoader
    #train_datasetからミニバッチごとに取り出すことを目的に使われる
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,   #how many samples per batch to load (default: 1).
        shuffle=True,  # set to True to have the data reshuffled at every epoch (default: False).
        drop_last=True,
        num_workers=8,   #ミニバッチを作成する際の並列実行数を指定できる.最大で CPU の論理スレッド数分の高速化が期待できる.
    )
    #valデータセットからミニバッチを作成する理由がいまいち分からない
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8
    )
    return train_loader, val_loader

########################################################################################################################
# train loop
########################################################################################################################


def main():
    parser = argparse.ArgumentParser()   #パーサを作る
    # parser.add_argumentで受け取る引数を追加していく
    parser.add_argument("--data_dir", required=True)  # オプション引数を追加,required=True:指定必須
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()  # 引数を解析

    data_dir = pathlib.Path(args.data_dir)
    train_dir = os.path.join(data_dir, "train")

    print(data_dir)   #パスオブジェクト
    print(type(data_dir))
    print(data_dir.exists())    #パスの存在を確認
    print(train_dir)

if __name__ == "__main__":
    main()
