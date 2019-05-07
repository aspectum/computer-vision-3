import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import req1, req2, req2_meu

def main():
    if (sys.argv[1] == "--req1") :
        req1.main()
    elif (sys.argv[1] == "--req2") :
        req2.main()
    elif (sys.argv[1] == "--req2_non_rect") :
        req2_meu.main()
    else:
        print("Chamada errada: ", sys.argv[1])
        print("Chamadas válidas: ")
        print("python src/pd3.py --req1: ")
        print("python src/pd3.py --req2: ")
        print("python src/pd3.py --req2_non_rect: ")

    


if __name__  == "__main__":
    main()