import os
import pdb
from datetime import datetime

def init_remote(opt):
    os.system(f"rm -rf output/{opt.name}")
    cwd = os.getcwd()
    os.system(f"gsutil cp -r {opt.save_remote_gs}/{opt.name} ./output/")
    if os.path.exists(f"output/{opt.name}/iter.txt") and not os.path.exists(f"checkpoints/{opt.name}/iter.txt"):
        os.system(f"cp output/{opt.name}/latest_net_*.pth checkpoints/{opt.name}/")
        os.system(f"cp output/{opt.name}/iter.txt checkpoints/{opt.name}/")

def upload_remote(opt):
    os.system(f"gsutil cp -r {opt.save_remote_gs}/{opt.name}/savemodel ./output/{opt.name}")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    os.system(f"cp output/{opt.name}.html output/{opt.name}/")
    os.system(f"cp checkpoints/{opt.name}/opt.txt output/{opt.name}/")
    os.system(f"cp checkpoints/{opt.name}/iter.txt output/{opt.name}/")
    os.system(f"echo {dt_string} > output/{opt.name}/time.txt")
    with open(f"output/{opt.name}/savemodel","r") as f:
        line = f.readlines()
    if line[0].startswith("y"):
        os.system(f"gsutil cp -r ./checkpoints/{opt.name}/latest_net_*.pth {opt.save_remote_gs}/{opt.name}/")
        with open(f"output/{opt.name}/savemodel", "w") as f:
            f.writelines("n")
    os.system(f"gsutil cp -r ./output/{opt.name} {opt.save_remote_gs}/")



if __name__ == "__main__":
    class Temp:
        pass
    opt = Temp()
    opt.save_remote_gs = "gs://zengxianyu"
    opt.name = "cline"
    init_remote(opt)
