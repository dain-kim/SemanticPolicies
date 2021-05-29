# Config for testscene3.ttt

CUP_ID_TO_NAME      = {21: 'red cup', 23: 'blue cup', 0: 'NONE'}
BIN_ID_TO_NAME      = {16: 'yellow dish', 17: 'red dish', 18: 'green dish', 0: 'NONE'}
SIM_CUP_TO_FRCNN    = {1:21, 3:23, 4:21, 6:23}
SIM_BIN_TO_FRCNN    = {1:16, 2:17, 3:18}

def mapObjectIDs(oid):
    if oid == 127: # red cup
        return 21
    elif oid == 128: # red cup
        return 21
    elif oid == 129: # green cup (not used)
        return 21
    elif oid == 130: # green cup (not used)
        return 23
    elif oid == 126: # blue cup
        return 23
    elif oid == 131: # blue cup
        return 23

    elif oid == 117: # yellow bin
        return 16
    elif oid == 116: # red bin
        return 17
    elif oid == 115: # green bin
        return 18
    elif oid == 114: # blue bin (not used)
        return 18
    elif oid == 113: # pink bin (not used)
        return 18

    elif oid == -1: # nothing is grabbed
        return 0
    
    else:
        print('unidentified object in mapOIDs')
        print(oid)
        return 0

