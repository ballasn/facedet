from utils.test_cascade import Cascade
import numpy as np

if __name__ == "__main__":
    c = Cascade()
    c.append("../../exp/train_16_700_best.pkl", 0.1)
    c.append("../../exp/2layers_16.pkl", 0.2)
    c.insert("../../exp/2layers_16.pkl", 0.0, 1)
    c.insert("../../exp/2layers_16.pkl", 0.4, 2)
    c.prepare_nms()
    c.prepare_nms()
    c.compile_fprops()
    c.compile_fprops()
    print c
    x = np.random.rand(3, 17, 17, 1)
    x = np.array(x, dtype=np.float32)
    print c.one_stage_fprop(x, 0)

