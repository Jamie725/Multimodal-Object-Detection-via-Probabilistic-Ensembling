import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression


inputdir = '/home/jack/MSCV_Capstone/fall/week18/trainset_svm_final'
X_train = []
Y_train = []
for filename in os.listdir(r"" + inputdir):
    file_dir = os.path.join(inputdir, filename)
    fo = open(file_dir, 'r')
    lines = fo.readlines()


    for line in lines:
        line = line.split()
        label,feature1,feature2,feature3,feature4 = int(line[1]), float(line[2]), float(line[3]), float(line[4]),float(line[5])
        #print(label,feature1,feature2,feature3,feature4)
        X_train.append([feature1,feature2,feature3,feature4])
        Y_train.append(label)

# print(len(X_train))
# print(len(Y_train))
clf = CalibratedClassifierCV(svm.SVC())
clf.fit(X_train, Y_train)
# y = clf.predict([[0.3618894, -0.31067058, 0, 0]])
# print(y)
# y_proba = clf.predict_proba([[0.3618894, -0.31067058, 0, 0]])
# print(y_proba)
# clf = LogisticRegression(C=1e5)
# clf.fit(X_train, Y_train)
# y = clf.predict([[0.3618894, -0.31067058, 0, 0]])
# print(y)
# y_proba = clf.predict_proba([[0.3618894, -0.31067058, 0, 0]])
# print(y_proba)

inputdir2 = '/home/jack/MSCV_Capstone/fall/week18/testset_svm'
save_dir = '/home/jack/MSCV_Capstone/fall/week18/svm_outprob_final/test'
if not os.path.exists(save_dir):  # if it doesn't exist already
    os.makedirs(save_dir)

for filename in os.listdir(r"" + inputdir2):
    print(filename)
    file_dir = os.path.join(inputdir2, filename)
    fw = open(os.path.join(save_dir, filename), 'w')
    fo = open(file_dir, 'r')
    lines = fo.readlines()

    for line in lines:
        line = line.split()

        feature1, feature2, feature3, feature4 = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        print(feature1, feature2, feature3, feature4)
        y = clf.predict([[feature1, feature2, feature3, feature4]])
        print('pred',y)
        y_proba = clf.predict_proba([[feature1, feature2, feature3, feature4]])
        x1,y1,x2,y2 = float(line[5]), float(line[6]), float(line[7]), float(line[8])
        print('prob',y_proba[0][1])
        fw.write(
            'person' +  ' ' + str(y_proba[0][1]) + " " + str(x1) + " " + str(y1) + " " + str(x2) + ' ' + str(y2))
        fw.write("\n")
    fw.close()
    fo.close()





