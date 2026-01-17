#!/usr/bin/env python3
# code_B_eval_all.py (INPUT_MODE + Lab対応版)
from __future__ import annotations
import argparse, math, re, cv2, yaml
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.image as mpimg

# ============================================================
# CLI 引数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="指定datasets/<DATASET>を使って lab_logs/expXX の best.pth を一括評価"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="valid", choices=["train","valid","test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_root", type=str, default="lab_logs")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--exp_from", type=str, default=None)
    parser.add_argument("--exp_to", type=str, default=None)
    return parser.parse_args()

def parse_exp_number(v: str|None):
    if v is None: return None
    m = re.search(r"(\d+)$", v.strip())
    return int(m.group(1)) if m else None

# ============================================================
# Math helpers
# ============================================================
def sincos2deg(v: torch.Tensor):
    rad = torch.atan2(v[...,0], v[...,1])
    return (rad * 180.0 / math.pi).remainder(360.0)

def circular_error(p, t):
    return np.abs((p - t + 180.0) % 360.0 - 180.0)

def _wrap_pi(x): return (x+math.pi)% (2*math.pi)-math.pi

# ============================================================
# Model
# ============================================================
class DilatedBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,stride=stride,padding=dilation,dilation=dilation,bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,padding=dilation,dilation=dilation,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out+identity)

class ResNet18Dilated(nn.Module):
    def __init__(self,in_ch=3,out_dim=2,width_mult=1.0,hidden_dim=256,dropout_p=0.3):
        super().__init__()
        base = np.array([64,128,256,512])
        chs = np.maximum(1,(base*width_mult).astype(int)).tolist()
        self.inplanes = chs[0]

        self.conv1 = nn.Conv2d(in_ch,chs[0],3,1,1,bias=False)
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_layer(chs[0],2,1,1)
        self.layer2 = self._make_layer(chs[1],2,2,2)
        self.layer3 = self._make_layer(chs[2],2,2,4)
        self.layer4 = self._make_layer(chs[3],2,2,8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[3],hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim,out_dim),
        )

    def _make_layer(self,planes,blocks,stride,dilation):
        down = None
        if stride!=1 or self.inplanes!=planes:
            down = nn.Sequential(
                nn.Conv2d(self.inplanes,planes,1,stride,bias=False),
                nn.BatchNorm2d(planes)
            )
        layers=[DilatedBasicBlock(self.inplanes,planes,stride,down,dilation)]
        self.inplanes=planes
        for _ in range(1,blocks):
            layers.append(DilatedBasicBlock(self.inplanes,planes,dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        x=self.avgpool(x)
        v = nn.functional.normalize(self.fc(x).view(x.size(0),-1,2),dim=2)
        return v.view(v.size(0),-1)

# ============================================================
# Dataset INPUT_MODE対応
# ============================================================
class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir:Path, img_size:int,
                 input_mode:str, color_mode:str="lab"):  ### ★ 修正
        self.img_dir = img_dir
        self.img_size = img_size
        self.input_mode = input_mode.lower()            ### ★ 修正
        self.color_mode = color_mode.lower()

        labels = self.img_dir.parent / "labels.csv"
        df = pd.read_csv(labels)
        self.paths = [self.img_dir / f for f in df["filename"]]
        self.targets = df["roll"].astype(np.float32).values

    def __len__(self): return len(self.paths)

    def __getitem__(self,idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_np = np.array(img.resize((self.img_size,self.img_size),Image.LANCZOS))

        if self.color_mode=="lab":
            lab = cv2.cvtColor(img_np,cv2.COLOR_RGB2LAB).astype(np.float32)

            if self.input_mode in ("gray","bin4"):
                L = lab[:,:,0:1]
                L = (L/255.0 - 0.5)/0.5
                tensor = torch.from_numpy(L.transpose(2,0,1))   # (1,H,W)
            else:
                lab = (lab/255.0 - 0.5)/0.5
                tensor = torch.from_numpy(lab.transpose(2,0,1)) # (3,H,W)
        else:
            tensor = transforms.ToTensor()(img)
            tensor = transforms.Normalize([0.5]*3,[0.5]*3)(tensor)

        return tensor, self.targets[idx], self.paths[idx].name

# ============================================================
# EKF（変わらず）
# ============================================================
KF_FPS=15.0; KF_SIGMA_Z_DEG=1.0; KF_SIGMA_A_DEG=2.0
KF_INIT_STD_TH_DEG=20.0; KF_INIT_STD_OM_DEG=10.0

class EKFRoll:
    def __init__(self,theta0,omega0=0.0):
        self.x=np.array([float(theta0),float(omega0)])
        self.P=np.diag([
            math.radians(KF_INIT_STD_TH_DEG)**2,
            math.radians(KF_INIT_STD_OM_DEG)**2
        ])
        self.R=np.eye(2)*(math.radians(KF_SIGMA_Z_DEG)**2)
        self.sigma_a=math.radians(KF_SIGMA_A_DEG)

    def step(self,z,dt):
        F=np.array([[1,dt],[0,1]])
        G=np.array([[0.5*dt**2],[dt]])
        self.x = F@self.x
        self.x[0]=_wrap_pi(self.x[0])
        self.P = F@self.P@F.T + (self.sigma_a**2)*(G@G.T)

        th=self.x[0]
        h=np.array([math.sin(th),math.cos(th)])
        H=np.array([[math.cos(th),0],[-math.sin(th),0]])

        S=H@self.P@H.T + self.R
        K=self.P@H.T@np.linalg.inv(S)
        self.x += K@(z-h)
        self.x[0]=_wrap_pi(self.x[0])
        self.P=(np.eye(2)-K@H)@self.P
        return self.x.copy()

# ============================================================
# 評価本体 INPUT_MODE対応
# ============================================================
def eval_one_exp(exp_dir:Path, args):
    cfg_path=exp_dir/"config_used.yaml"
    ckpt=exp_dir/"checkpoints"/"best.pth"
    if not cfg_path.exists() or not ckpt.exists():
        print("[SKIP]", exp_dir.name); return

    cfg=yaml.safe_load(open(cfg_path))
    input_mode=cfg.get("INPUT_MODE","rgb")     ### ★ 修正
    color_mode=cfg.get("COLOR_MODE","lab")
    img_size=cfg.get("IMG_SIZE",[224,224])[0]
    resize_mode=cfg.get("RESIZE_MODE","area")
    model_name=cfg.get("MODEL_NAME","ResNet18Dilated_1p0")

    if "_" in model_name:
        width=float(model_name.split("_")[-1].replace("p","."))
    else: width=1.0

    in_ch = 1 if input_mode.lower() in ("gray","bin4") else 3   ### ★ 修正

    model = ResNet18Dilated(in_ch=in_ch, out_dim=2,
                            width_mult=width,
                            hidden_dim=cfg.get("HIDDEN_DIM",256),
                            dropout_p=cfg.get("DROPOUT_P",0.3)).to("cuda" if torch.cuda.is_available() else "cpu")

    print("  Load:",ckpt)
    state=torch.load(ckpt,map_location="cpu")
    model.load_state_dict(state); model.eval()

    # ★ DATASET_PATHをINPUT_MODEに従って構成
    test_path = (Path(args.dataset_root)
        / args.dataset
        / "cache"
        / input_mode              ### ★ 修正
        / f"sz{img_size}_{resize_mode}"
        / args.split
        / "imgs"
    )

    if not test_path.exists():
        print("[SKIP] test_path missing:",test_path)
        return

    ds = ImageRegressionDataset(test_path,img_size,input_mode,color_mode)  ### ★ 修正

    ld = DataLoader(ds,batch_size=args.batch_size,
                    shuffle=False,num_workers=args.num_workers)

    device=next(model.parameters()).device
    results=[]; sincos_list=[]
    print(f"  Eval {args.dataset}/{args.split} ({len(ds)} samples)")
    with torch.no_grad():
        for imgs,tgts,names in tqdm(ld, desc=f"{exp_dir.name}"):
            imgs=imgs.to(device)
            outs=model(imgs).cpu()
            preds=sincos2deg(outs).numpy()
            tgts_np=tgts.numpy()
            sincos_list.extend(outs.numpy())

            for n,t,p in zip(names,tgts_np,preds):
                results.append({
                    "filename":n,
                    "true_roll":float(t),
                    "pred_roll":float(p),
                    "err_roll":float(circular_error(p,t)),
                })

    df=pd.DataFrame(results)
    if len(df)==0: return

    # === EKF ===
    df_kf=df.sort_values("filename").reset_index(drop=True)
    theta0=math.atan2(sincos_list[0][0],sincos_list[0][1])
    kf=EKFRoll(theta0)
    kf_res=[]
    for i,row in df_kf.iterrows():
        idx=df[df["filename"]==row["filename"]].index[0]
        st=kf.step(np.array(sincos_list[idx]),1.0/KF_FPS)
        th=math.degrees(st[0])%360
        kf_res.append({"filename":row["filename"],
                       "kf_roll":th,
                       "kf_err":float(circular_error(th,row["true_roll"]))})

    df=df.merge(pd.DataFrame(kf_res),on="filename")

    out_dir=exp_dir/f"eval_{args.dataset}_{args.split}_kf"
    figs=out_dir/"figs"
    figs.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_dir/"eval_results_kf.csv",index=False)
    print("  Saved:",out_dir/"eval_results_kf.csv")

    # === 可視化 ===
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(df["true_roll"],df["err_roll"],s=8,alpha=0.5,label="Raw")
    ax.scatter(df["true_roll"],df["kf_err"],s=8,alpha=0.5,label="KF")
    ax.set_xlabel("True Deg"); ax.set_ylabel("Error Deg"); ax.legend()
    fig.savefig(figs/"error_vs_true.png"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5))
    ax.hist(df["err_roll"],bins=30,alpha=0.5,label=f"Raw(mean={df['err_roll'].mean():.2f})")
    ax.hist(df["kf_err"],bins=30,alpha=0.5,label=f"KF(mean={df['kf_err'].mean():.2f})")
    ax.legend()
    fig.savefig(figs/"error_hist.png"); plt.close(fig)

    print("  Done:",exp_dir.name)

# ============================================================
# MAIN
# ============================================================
def main():
    args=parse_args()
    root=Path(args.log_root)
    if not root.exists(): print("Not found:",root); return

    f=parse_exp_number(args.exp_from)
    t=parse_exp_number(args.exp_to)

    exp_dirs=[]
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("exp"):
            m=re.match(r"exp(\d+)",p.name); 
            if not m: continue
            n=int(m.group(1))
            if f is not None and n<f: continue
            if t is not None and n>t: continue
            exp_dirs.append(p)

    if not exp_dirs: print("No matching exp dirs"); return
    print("Found",len(exp_dirs),"experiments")

    for p in exp_dirs:
        print("\n===",p.name,"===")
        try: eval_one_exp(p,args)
        except Exception as e: print(" [ERROR]",e)

if __name__=="__main__":
    main()
