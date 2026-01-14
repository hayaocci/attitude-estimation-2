#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_kf.py
- eval_results.csv（または同等のCSV）から 生MAE と KF後MAE を計算
- σz（観測ノイズ）, σa（角加速度ノイズ）, fps（= 1/dt）をグリッドサーチ最適化
- 最良パラメータでの KF 軌跡を CSV 出力（<input_dir>/eval_results_kf_optimized.csv）

想定CSVカラム:
  filename,true_roll,pred_roll  （順不同OK。必要に応じて --true-col 等で指定）

使い方例:
  python optimize_kf.py --csv path/to/eval_results.csv
  python optimize_kf.py --csv eval_results.csv --fps 30 60 --sigma-z 1 2 3 5 --sigma-a 5 10 15
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------- 角度ユーティリティ ----------------
def wrap_deg_pm180(x: float) -> float:
    """[-180,180) へ折り畳み"""
    return (x + 180.0) % 360.0 - 180.0

def circ_abs_err(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """円周誤差の絶対値 [deg]（要素ごと）"""
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(diff)

def deg2rad(x: float | np.ndarray) -> float | np.ndarray:
    return (np.pi / 180.0) * x

def rad2deg(x: float | np.ndarray) -> float | np.ndarray:
    return (180.0 / np.pi) * x

def wrap_pi(x: float) -> float:
    """[-pi, pi)"""
    return (x + np.pi) % (2*np.pi) - np.pi

# ---------------- EKF 実装（観測 = [sinθ, cosθ]） ----------------
class EKFRoll:
    """
    状態 x=[θ, ω], 観測 z=[sin(θ), cos(θ)]
    - θ [rad], ω [rad/s]
    - Q は角加速度白色ノイズを積分した等価
    """
    def __init__(self,
                 theta0: float,
                 omega0: float = 0.0,
                 sigma_theta0: float = deg2rad(20.0),
                 sigma_omega0: float = deg2rad(10.0),
                 sigma_a: float = deg2rad(5.0),     # [rad/s^2]
                 sigma_z: float = deg2rad(5.0)):    # 観測由来角度ノイズ -> (sin,cos)空間に等方設定
        self.x = np.array([theta0, omega0], dtype=float)
        self.P = np.diag([sigma_theta0**2, sigma_omega0**2])
        self.sigma_a = float(sigma_a)
        self.R = np.eye(2) * (sigma_z**2)

    def predict(self, dt: float):
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        G = np.array([[0.5*dt**2],
                      [dt      ]], dtype=float)
        Q = (self.sigma_a**2) * (G @ G.T)
        self.x = F @ self.x
        self.x[0] = wrap_pi(self.x[0])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray):
        th = self.x[0]
        h  = np.array([np.sin(th), np.cos(th)], dtype=float)
        H  = np.array([[ np.cos(th), 0.0],
                       [-np.sin(th), 0.0]], dtype=float)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - h
        self.x = self.x + K @ y
        self.x[0] = wrap_pi(self.x[0])
        self.P = (np.eye(2) - K @ H) @ self.P

    def step(self, z: np.ndarray, dt: float):
        self.predict(dt)
        self.update(z)
        return self.x.copy()  # [theta, omega]

# ---------------- KF 一括実行 ----------------
def run_kf_on_series(
    true_deg: np.ndarray,
    pred_deg: np.ndarray,
    fps: float,
    sigma_z_deg: float,
    sigma_a_deg: float,
    init_std_th_deg: float = 20.0,
    init_std_om_deg: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    入力:
      true_deg, pred_deg : 角度列 [deg]（同じ長さ、時系列順）
      fps                : フレームレート（= 1/dt）
      sigma_z_deg        : 観測ノイズ（角度換算）[deg]
      sigma_a_deg        : 角加速度ノイズ [deg/s^2]
    出力:
      kf_deg             : KF 推定角 [deg]
      kf_mae             : 円周MAE [deg]
    """
    assert len(true_deg) == len(pred_deg) and len(true_deg) > 0
    dt = 1.0 / max(1e-6, float(fps))

    # 観測ベクトル (sin,cos) は pred から生成（モデル出力がL2正規化済み前提）
    obs = np.stack([np.sin(deg2rad(pred_deg)), np.cos(deg2rad(pred_deg))], axis=1)

    # 初期化
    theta0 = math.atan2(obs[0,0], obs[0,1])  # atan2(sin, cos)
    kf = EKFRoll(
        theta0=theta0,
        omega0=0.0,
        sigma_theta0=deg2rad(init_std_th_deg),
        sigma_omega0=deg2rad(init_std_om_deg),
        sigma_a=deg2rad(sigma_a_deg),
        sigma_z=deg2rad(sigma_z_deg),
    )

    out = np.zeros_like(true_deg, dtype=float)
    for i in range(len(true_deg)):
        th, om = kf.step(obs[i], dt)
        out[i] = (rad2deg(th)) % 360.0

    kf_mae = float(np.mean(circ_abs_err(out, true_deg)))
    return out, kf_mae

# ---------------- グリッドサーチ ----------------
def grid_search_kf(
    true_deg: np.ndarray,
    pred_deg: np.ndarray,
    fps_list: List[float],
    sigma_z_list: List[float],
    sigma_a_list: List[float],
    init_std_th_deg: float = 20.0,
    init_std_om_deg: float = 10.0,
) -> Tuple[Dict, pd.DataFrame]:
    """
    すべての組合せで KF を走らせ、MAE を表にして返す。
    戻り値:
      best: {"fps":..., "sigma_z":..., "sigma_a":..., "kf_mae":..., "raw_mae":...}
      table: 各組合せの結果 DataFrame（ソート済み）
    """
    raw_mae = float(np.mean(circ_abs_err(pred_deg, true_deg)))

    rows = []
    best = {"kf_mae": float("inf")}
    for fps in fps_list:
        for sz in sigma_z_list:
            for sa in sigma_a_list:
                _, kf_mae = run_kf_on_series(
                    true_deg=true_deg,
                    pred_deg=pred_deg,
                    fps=fps,
                    sigma_z_deg=sz,
                    sigma_a_deg=sa,
                    init_std_th_deg=init_std_th_deg,
                    init_std_om_deg=init_std_om_deg,
                )
                rows.append({"fps": fps, "sigma_z": sz, "sigma_a": sa,
                             "kf_mae": kf_mae, "raw_mae": raw_mae,
                             "improvement": raw_mae - kf_mae})
                if kf_mae < best["kf_mae"]:
                    best = {"fps": fps, "sigma_z": sz, "sigma_a": sa,
                            "kf_mae": kf_mae, "raw_mae": raw_mae,
                            "improvement": raw_mae - kf_mae}
    table = pd.DataFrame(rows).sort_values(["kf_mae", "fps", "sigma_z", "sigma_a"]).reset_index(drop=True)
    return best, table

# ---------------- メイン ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="eval_results.csv へのパス")
    ap.add_argument("--filename-col", default="filename")
    ap.add_argument("--true-col", default="true_roll")
    ap.add_argument("--pred-col", default="pred_roll")
    ap.add_argument("--sort-by", default="filename", help="時系列順のキー（例: filename）")
    # 検索グリッド
    ap.add_argument("--fps", type=float, nargs="+", default=[15.0, 30.0, 60.0])
    ap.add_argument("--sigma-z", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 7.0])
    ap.add_argument("--sigma-a", type=float, nargs="+", default=[2.0, 5.0, 10.0, 15.0, 25.0])
    # 初期分散（必要あれば変更）
    ap.add_argument("--init-std-theta", type=float, default=20.0)
    ap.add_argument("--init-std-omega", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by).reset_index(drop=True)

    true_deg = df[args.true_col].astype(float).values
    pred_deg = df[args.pred_col].astype(float).values

    # グリッドサーチ
    best, table = grid_search_kf(
        true_deg=true_deg,
        pred_deg=pred_deg,
        fps_list=args.fps,
        sigma_z_list=args.sigma_z,
        sigma_a_list=args.sigma_a,
        init_std_th_deg=args.init_std_theta,
        init_std_om_deg=args.init_std_omega,
    )

    # 表示
    print("\n=== Raw vs. Best KF ===")
    print(f"Raw   MAE: {best['raw_mae']:.4f} deg")
    print(f"Best  MAE: {best['kf_mae']:.4f} deg  "
          f"(improvement {best['improvement']:+.4f} deg)")
    print(f"Best params: fps={best['fps']}, sigma_z={best['sigma_z']} deg, sigma_a={best['sigma_a']} deg/s^2")

    # 保存
    out_dir = args.csv.parent
    table_path = out_dir / "kf_grid_search.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved grid table → {table_path}")

    # ベスト設定で KF をもう一度走らせて、CSVを保存
    best_kf_deg, _ = run_kf_on_series(
        true_deg=true_deg,
        pred_deg=pred_deg,
        fps=best["fps"],
        sigma_z_deg=best["sigma_z"],
        sigma_a_deg=best["sigma_a"],
        init_std_th_deg=args.init_std_theta,
        init_std_om_deg=args.init_std_omega,
    )
    df_best = df.copy()
    df_best["kf_roll_opt"] = best_kf_deg
    df_best["kf_err_opt"]  = circ_abs_err(best_kf_deg, true_deg)
    out_csv = out_dir / "eval_results_kf_optimized.csv"
    df_best.to_csv(out_csv, index=False)
    print(f"Saved best KF track → {out_csv}")

if __name__ == "__main__":
    main()
