import os, json, cv2
import torch
import numpy as np

class TerrariaDataLoader:
    def __init__(self, datapath, frame_dir):
        self.frame_dir = frame_dir
        self.data = []

        with open(datapath, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        item = self.data[ind]

        img = cv2.imread(os.path.join(self.frame_dir, item["frame"]))
        h, w, c = img.shape
        x_norm = max(0, min(1, (item["mouse"]["x"] / w)))
        y_norm = max(0, min(1, (item["mouse"]["y"] / h)))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        k = item["keys"]
        m = item["mouse"]

        action = [
            k["move_left"], k["move_right"], 
            k["move_up"], k["move_down"], 
            k["jump"], k["run"], 

            k["slot_1"], k["slot_2"], 
            k["slot_3"], k["slot_4"], 
            k["slot_5"], k["slot_6"], 
            k["slot_7"], k["slot_8"], 
            k["slot_9"], 

            k["heal"], k["grapple"], k["esc"], m["left"], m["right"],
            x_norm, y_norm
        ]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32)
        )
