# -*- coding: utf-8 -*-
# File: package/manage/file_manager.py

import os

def get_folder_names(dir_path: str) -> list[str]:
    """
    指定されたディレクトリ内のすべてのフォルダ名を取得する。

    Parameters:
        dir_path (str): フォルダ名を取得するディレクトリのパス。

    Returns:
        list[str]: ディレクトリ内のフォルダ名のリスト。
    """
    try:
        folder_names = [
            name for name in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, name))
        ]
        return folder_names
    except FileNotFoundError:
        print(f"Error: The directory '{dir_path}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_train_folder_numbers(folder_names: list[str], given_name: str = "train") -> list[int]:
    """
    フォルダ名のリストから、given_name が先頭に含まれる名前を抽出し、
    given_name を除去した末尾の数値を整数として返す。

    Parameters:
        folder_names (list[str]): フォルダ名のリスト。
        given_name (str): フィルタ対象の接頭辞（デフォルトは "train"）。

    Returns:
        list[int]: given_name を除いた整数番号のリスト（昇順）。
    """
    result = []
    for name in folder_names:
        if name.startswith(given_name):
            suffix = name[len(given_name):]
            if suffix.isdigit():
                result.append(int(suffix))
            else:
                print(f"Warning: '{name}' から '{suffix}' を整数に変換できませんでした。無視します。")
    return sorted(result)

def find_missing_or_next_number(numbers: list[int]) -> int:
    """
    昇順の整数リストから、欠けている最小の番号を返す。
    ・リストが1から始まっていない場合は1を返す。
    ・欠番がある場合は最小の欠番を返す。
    ・欠番がない場合は最大値 + 1 を返す。

    Parameters:
        numbers (list[int]): 昇順に並んだ整数リスト。

    Returns:
        int: 最小の欠番、または最大値 + 1、または1。
    """
    if not numbers or numbers[0] != 1:
        return 1

    for expected in range(1, len(numbers) + 1):
        if numbers[expected - 1] != expected:
            return expected

    return len(numbers) + 1

def make_new_log_folder(dir_path: str, given_name: str = "train", new_number: int = None) -> str:
    """
    新しいログフォルダを作成し、その中に "weights" フォルダも作成する。
    フォルダ名は given_name + 数字 の形式で、数字は欠番または最大値 + 1。

    Parameters:
        dir_path (str): フォルダを作成するディレクトリのパス。
        given_name (str): フォルダ名の接頭辞（デフォルトは "train"）。
        new_number (int): 指定された番号でフォルダを作成する場合に使用。

    Returns:
        str: 作成されたメインフォルダのパス。
    """
    folder_names = get_folder_names(dir_path)
    train_numbers = get_train_folder_numbers(folder_names, given_name)

    if new_number is None:
        new_number = find_missing_or_next_number(train_numbers)

    new_folder_name = f"{given_name}{new_number}"
    new_folder_path = os.path.join(dir_path, new_folder_name)

    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Created new folder: {new_folder_path}")

        # "weights" フォルダの作成
        weights_folder_path = os.path.join(new_folder_path, "weights")
        os.makedirs(weights_folder_path, exist_ok=True)
        print(f"Created 'weights' folder: {weights_folder_path}")

    except Exception as e:
        print(f"Error while creating folders: {e}")

    return new_folder_path


if __name__ == "__main__":
    test_dir = "C:/workspace/Github/attitude-estimation/log"

    folder_names = get_folder_names(test_dir)
    print("All folders:", folder_names)

    train_numbers = get_train_folder_numbers(folder_names)
    print("Filtered train numbers (int):", train_numbers)

    missing_number = find_missing_or_next_number(train_numbers)
    print(f"Missing or next number: {missing_number}")

    new_folder_path = make_new_log_folder(test_dir, "train", new_number=missing_number)
    print(f"New folder created (or already exists): {new_folder_path}")
