# savef.py の全内容をこちらに置き換えてください

import os
from datetime import datetime

def create_dir(dir_name=None):
    """
    結果を保存するディレクトリを作成する関数。
    引数 dir_name が指定されればその名前で、なければ現在時刻で作成する。
    """
    # 引数でディレクトリ名が指定されなかった場合、現在時刻を名前にする
    if dir_name is None:
        dir_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存先パスを生成
    path = './Result/' + dir_name
    
    # ディレクトリが存在しない場合は作成
    if not os.path.isdir(path):
        os.makedirs(path)
    
    return path

def make_doc(path, **kwargs):
    """
    シミュレーションのパラメータを記録するテキストファイルを作成する関数。
    """
    file_path = path + '/simulation_parameters.txt'
    with open(file_path, 'w') as f:
        f.write("--- Simulation Parameters ---\n")
        # 実行された時点でのパラメータを動的に書き出す
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Timestamp: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"パラメータファイルを作成しました: {file_path}")