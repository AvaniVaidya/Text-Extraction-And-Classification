import pickle
fileIndexArr=[0,0,0]
with open('fileIndex.pickle', 'wb') as b:
    pickle.dump(fileIndexArr,b)