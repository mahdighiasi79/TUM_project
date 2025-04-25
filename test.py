import time
import data_generators as dg


if __name__ == '__main__':
    cut_v = dg.Cut_V(5, 5, "s_e")
    t1 = time.time()
    print(cut_v.generate_data(1000000))
    t2 = time.time()
    print(t2-t1)
