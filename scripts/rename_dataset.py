import shutil
from pathlib import Path

# === 設定 ===
src_dir = Path("../img")   # 元画像フォルダ
dst_dir = Path("renamed_images") # 出力フォルダ（自動作成される）
offset = 36                      # 加算する数

dst_dir.mkdir(exist_ok=True)

# === 変換処理 ===
for src_path in sorted(src_dir.glob("*.png")):
    stem = src_path.stem  # "0000" の部分
    try:
        num = int(stem)   # 数値に変換できる場合のみ処理
    except ValueError:
        print(f"スキップ: {src_path.name}（数値名ではない）")
        continue

    new_num = num + offset
    new_name = f"{new_num:04d}.png"  # 4桁ゼロ埋め
    dst_path = dst_dir / new_name

    print(f"{src_path.name} -> {new_name}")
    shutil.copy2(src_path, dst_path)  # 元は残しつつコピー
