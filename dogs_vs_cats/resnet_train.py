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



# trainデータをtrain_val_split
def setup_train_val_split(labels, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)
    #
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, random_state=seed
    )
    #
    train_indices, val_indices = next(splitter.split(x, y))
    #
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

#
def setup_train_val_datasets(data_dir, dryrun=False):
    #torchvision.datasets.ImageFolderを使用してデータセットの作成
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train"),   #trainデータの入っているディレクトリのパス
        transform=setup_center_crop_transform()   #transformの指定
    )
    # labels=正解ラベルの配列
    labels = get_labels(dataset)
    #
    train_indices, val_indices = setup_train_val_split(labels, dryrun)

    #訓練セットと検証セットにデータセットを分割する際はSubsetクラスを用いる
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)


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
