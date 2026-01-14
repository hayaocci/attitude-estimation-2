# 任意のディレクトリに存在するcsvファイルをすべて読み込み、1つのグラフを作成する。

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_train_graph(dir_path: str): 
    """
    任意のディレクトリに存在する"train_log.csv"を読み込み、学習の進捗をグラフ化する。
    """
    file_path = os.path.join(dir_path, "train_log.csv")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    output_path = os.path.join(dir_path, "train_progress.png")

    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
    plt.plot(df['epoch'], df['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid()

    # 保存する
    plt.savefig(output_path)

    plt.show()
    plt.close()

def make_valid_graph(dir_path: str):
    """
    任意のディレクトリに存在する"valid_log.csv"を読み込み、検証の進捗をグラフ化する。
    """
    file_path = os.path.join(dir_path, "valid_log.csv")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    output_path = os.path.join(dir_path, "valid_progress.png")

    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Progress')
    plt.legend()
    plt.grid()

    # 保存する
    plt.savefig(output_path)

    plt.show()
    plt.close()

if __name__ == "__main__":
    # 例として、現在のディレクトリを指定
    test_dir = "C:/workspace/Github/attitude-estimation/log/train1"
    make_train_graph(test_dir)
    # make_graph_from_csv(test_dir)
