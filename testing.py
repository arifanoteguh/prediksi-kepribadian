# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testing.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import cv2
import math
import numpy as np
import numpycnn
import time
import progressbar

class Ui_Testing(object):

    def segmentasi(self, thres_img):
        contours, _ = cv2.findContours(thres_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        boxes = np.asarray(boxes)
        # need an extra "min/max" for contours outside the frame
        left = np.min(boxes[:, 0])
        top = np.min(boxes[:, 1])
        right = np.max(boxes[:, 2])
        bottom = np.max(boxes[:, 3])

        cv2.rectangle(thres_img, (left, top), (right, bottom), (255, 0, 0), 2)
        crop_img = thres_img[np.min(boxes[:, 1]):np.max(boxes[:, 3]), np.min(boxes[:, 0]):np.max(boxes[:, 2])]
        return crop_img

    def resize(self, image):
        crop_img = self.segmentasi(image)
        pa = crop_img.shape[0]
        la = crop_img.shape[1]

        resize_img = np.zeros(shape=[300, 300], dtype=np.uint8)
        for i in range(0, pa - 1):
            for j in range(0, la - 1):
                if (crop_img[i, j] == 255):
                    x = math.floor((300 * i) / pa)
                    y = math.floor((300 * j) / la)
                    resize_img[x, y] = crop_img[i, j]

        return resize_img

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "Pilih Tulisan Tangan", "", "Image Files (*.JPG)", options=options)
        if fileName:
            img = cv2.imread(fileName)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
            img_crop = self.resize(img_thresh)
            self.testingCNN(img_crop)
            cv2.imshow("Tulisan Tangan", img)

    def testingCNN(self,img_test):
        print("Proses")
        bar = progressbar.ProgressBar(maxval=2, widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
        bar.start()

        ## Panggil filter 12345
        filter_load = np.load("bobot/filter.npz")
        l1_filter = filter_load['f1']
        l2_filter = filter_load['f2']
        # l3_filter = filter_load['f3']
        # l4_filter = filter_load['f4']
        # l5_filter = filter_load['f5']

        ## Panggil bobot+bias
        bobot_simpan = np.load("bobot/0,000125/bobot200.npz")

        weight = bobot_simpan['bobot']
        bias = bobot_simpan['biass']

        ## Layer 1 ##
         # C1 #
        l1_feature_map = numpycnn.conv(img_test,l1_filter)
        l1_feature_map_relu = numpycnn.relu(l1_feature_map)

         # S1 #
        l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu)

        bar.update(1)
        time.sleep(0.1)

        ## Layer 2 ##
         # C2 #
        l2_feature_map = numpycnn.conv(l1_feature_map_relu_pool, l2_filter)
        l2_feature_map_relu = numpycnn.relu(l2_feature_map)

         # S2 #
        l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu)

        bar.update(2)
        time.sleep(0.1)

        # ## Layer 3 ##
        #  # C3 #
        # l3_feature_map = numpycnn.conv(l2_feature_map_relu_pool, l3_filter)
        # l3_feature_map_relu = numpycnn.relu(l3_feature_map)
        #
        #  # S3 #
        # l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu)
        #
        # bar.update(3)
        # time.sleep(0.1)
        #
        # ## Layer 4 ##
        #  # C4 #
        # l4_feature_map = numpycnn.conv(l3_feature_map_relu_pool, l4_filter)
        # l4_feature_map_relu = numpycnn.relu(l4_feature_map)
        #
        #  # S4 #
        # l4_feature_map_relu_pool = numpycnn.pooling(l4_feature_map_relu)
        #
        # bar.update(4)
        # time.sleep(0.1)
        #
        # ## Layer 5 ##
        #  # C5 #
        # l5_feature_map = numpycnn.conv(l4_feature_map_relu_pool, l5_filter)
        # l5_feature_map_relu = numpycnn.relu(l5_feature_map)
        # npad = ((0,1), (0,1), (0,0))
        # l5_feature_map_relu = np.pad(l5_feature_map_relu, pad_width=npad, mode='constant', constant_values=0)
        #
        #  # S5 #
        # l5_feature_map_relu_pool = numpycnn.pooling(l5_feature_map_relu)
        #
        # bar.update(5)
        # time.sleep(0.1)

        bar.finish()

        ### Vektorisasi ###
        vektor = np.zeros((4,1,5184))
        for i in range(0,4):
            vektor[i,:,:] = l2_feature_map_relu_pool[:,:,i].flatten()

        flatten = np.reshape(vektor, 20736).flatten()
        flatten = flatten.reshape((-1,1))

        ### Fully-connected ###
        kelas = ["Optimis","Pesimis","Seimbang"]

        fully = []
        for i in range(0, len(kelas)):
            fully.append(np.sum(flatten * weight[i, :]) + bias[i])

        ## Exponensial ##
        eksponen = []
        for i in range(0, len(kelas)):
            eksponen.append(math.exp(fully[i]))

        ## Softmax ##
        softmax = []
        for i in range(0, len(kelas)):
            softmax.append(eksponen[i] / sum(eksponen))

        print("\n")
        # print("--> Hasil prediksi :",softmax)
        print("Maka, kepribadian yang diprediksi adalah :",kelas[np.argmax(softmax)])
        self.label.setText(kelas[np.argmax(softmax)])

    def setupUi(self, Testing):
        Testing.setObjectName("Testing")
        Testing.resize(311, 252)
        self.centralwidget = QtWidgets.QWidget(Testing)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 120, 291, 111))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(10, 30, 621, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.btn_Upload = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Upload.setGeometry(QtCore.QRect(100, 50, 121, 51))
        self.btn_Upload.setObjectName("btn_Upload")
        Testing.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Testing)
        self.statusbar.setObjectName("statusbar")
        Testing.setStatusBar(self.statusbar)

        self.btn_Upload.clicked.connect(self.openFile)

        self.retranslateUi(Testing)
        QtCore.QMetaObject.connectSlotsByName(Testing)

    def retranslateUi(self, Testing):
        _translate = QtCore.QCoreApplication.translate
        Testing.setWindowTitle(_translate("Testing", "Training - Prediksi Kepribadian"))
        self.groupBox_2.setTitle(_translate("Testing", "Hasil Prediksi Kepribadian Dengan CNN"))
        self.label.setText(_translate("Testing", ""))
        self.label_2.setText(_translate("Testing", "Testing CNN"))
        self.btn_Upload.setText(_translate("Testing", "Pilih Tulisan Tangan"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Testing = QtWidgets.QMainWindow()
    ui = Ui_Testing()
    ui.setupUi(Testing)
    Testing.show()
    sys.exit(app.exec_())

