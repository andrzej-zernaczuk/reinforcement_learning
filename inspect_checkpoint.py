import pickle
with open("checkpoints/doubleq_r0.pkl","rb") as f:
    ckpt = pickle.load(f)
ckpt.keys()
