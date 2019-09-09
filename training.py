# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'training_cnn_baru.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import numpycnn
import numpy as np
import os
import cv2
import math
import time
import random
import progressbar


class Ui_TrainingCNN(object):

    #### Fungsi Prepro Grayscale ####
    def grayscale(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray_img

    #### END ####

    #### Fungsi Prepro Threshold ####
    def threshold(self, gray_img):
        _, thres_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        return thres_img

    #### END ####

    #### Fungsi Prepro Segmentasi ###
    def segmentasi(self, thres_img, img):
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

        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        crop_img = thres_img[np.min(boxes[:, 1]):np.max(boxes[:, 3]), np.min(boxes[:, 0]):np.max(boxes[:, 2])]

        return crop_img

    #### END ###

    #### Fungsi Prepro Resize ###
    def resize(self, crop_img):

        pa = crop_img.shape[0]
        la = crop_img.shape[1]

        resize_img = np.zeros(shape=[300, 300], dtype=np.uint8)
        for i in range(0, pa):
            for j in range(0, la):
                if (crop_img[i, j] == 255):
                    x = math.floor((300 * i) / pa)
                    y = math.floor((300 * j) / la)
                    resize_img[x, y] = crop_img[i, j]

        for i in range(0, resize_img.shape[0]):
            for j in range(0, resize_img.shape[1]):
                resize_img[i, j] = resize_img[i, j]/255

        return resize_img

    #### END ###

    #### Fungsi Training CNN ###
    def trainingCNN(self):
        ## Timer ##
        start = time.time()

        # ##### Siapin Data #####
        training_data = []
        DATADIR = "D:/Datasets/Benar/TulisanTanganTest/"
        CATEGORIES = ["Optimis", "Pesimis", "Seimbang"]

        list1 = os.listdir(os.path.join(DATADIR, "Optimis"))
        list2 = os.listdir(os.path.join(DATADIR, "Pesimis"))
        list3 = os.listdir(os.path.join(DATADIR, "Seimbang"))
        jumlah_file = len(list1) + len(list2) + len(list3)

        print("Preprocessing")
        print(jumlah_file)
        bar = progressbar.ProgressBar(maxval=50,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        x=0
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)  # Ambil Folder Kelas['Optimis','Pesimis','Seimbang']
            class_num = CATEGORIES.index(category)
            for gambar in os.listdir(path):
                img = cv2.imread(os.path.join(path,gambar))
                gray = self.grayscale(img)
                thres = self.threshold(gray)
                crop = self.segmentasi(thres,img)
                res = self.resize(crop)

                training_data.append([res, class_num])

                random.shuffle(training_data)

                bar.update(x + 1)
                time.sleep(0.1)
                x=x+1

        bar.finish()

        end = time.time()
        cetak = "Lamanya Proses : "+str(int(round(end - start)))+"s"
        self.textBrowser.append(cetak)

        # Split data training
        X = []
        y = []
        for features, label in training_data:
            X.append(features)
            y.append(label)

        for i in range(0,len(y)):
            ohl = [0,0,0]
            ohl[y[i]] = 1
            y[i] = ohl

        y = np.array(y)


        # np.savez("bobot/draft/data.npz", matriks=X, label=y)

        # data_load = np.load("bobot/data.npz")
        # X = data_load['matriks']
        # y = data_load['label']

        # print(y)
        ##### END #####

        ##### CNN #####

        #### Inisialisasi Awal ####

        ## Panggil bobot awal

        ## Panggil filter 12345
        filter_load = np.load("bobot/filter.npz")
        l1_filter = filter_load['f1']
        l2_filter = filter_load['f2']
        # l3_filter = filter_load['f3']
        # l4_filter = filter_load['f4']
        # l5_filter = filter_load['f5']

        ## Panggil weight
        # weight = np.load("bobot/weight_awal.npy")

        weight = np.zeros((3, 20736, 1))
        for i in range(0, 2):
            for j in range(0, 20735):
                angka = round(random.uniform(-0.05, 0.05), 2)
                weight[i, j, :] = angka

        ## Panggil Flatten
        flatten = np.load("bobot/flatten.npy")

        #### Feedforward ####
        print("Training CNN")
        # bar.start()
        x = 0
        bias = [0, 0, 0]
        epoch = 200
        lrate = 0.000125

        for ep in range(0, epoch):
            for image_i in range(0, len(X)):

                cetak = "Input Ke-" + str(image_i + 1)
                print(cetak)
                self.textBrowser.append(cetak)

                ### Layer 1 ###
                # C1
                l1_feature_map = numpycnn.conv(X[image_i],l1_filter)
                l1_feature_map_relu = numpycnn.relu(l1_feature_map)

                # S1
                l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu)

                # ### Layer 2 ###
                # C2
                l2_feature_map = numpycnn.conv(l1_feature_map_relu_pool, l2_filter)
                l2_feature_map_relu = numpycnn.relu(l2_feature_map)

                # S2
                l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu)

                # ### Layer 3 ###
                # # C3
                # l3_feature_map = numpycnn.conv(l2_feature_map_relu_pool, l3_filter)
                # l3_feature_map_relu = numpycnn.relu(l3_feature_map)
                #
                # # S3
                # l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu)
                #
                # ### Layer 4 ###
                # # C4
                # l4_feature_map = numpycnn.conv(l3_feature_map_relu_pool, l4_filter)
                # l4_feature_map_relu = numpycnn.relu(l4_feature_map)
                #
                # # S4
                # l4_feature_map_relu_pool = numpycnn.pooling(l4_feature_map_relu)
                #
                # ### Layer 5 ###
                # # C5
                # l5_feature_map = numpycnn.conv(l4_feature_map_relu_pool, l5_filter)
                # l5_feature_map_relu = numpycnn.relu(l5_feature_map)
                # npad = ((0,1), (0,1), (0,0))
                # l5_feature_map_relu = np.pad(l5_feature_map_relu, pad_width=npad, mode='constant', constant_values=0)
                #
                # # S5
                # l5_feature_map_relu_pool = numpycnn.pooling(l5_feature_map_relu)
                #
                ### Vektorisasi ###
                vektor = np.zeros((4,1,5184))
                for i in range(0,4):
                    vektor[i,:,:] = l2_feature_map_relu_pool[:,:,i].flatten()

                flatten = np.reshape(vektor, 20736).flatten()
                flatten = flatten.reshape((-1,1))

                flatten_simpan.append(flatten)

                ### Fully-connected ###
                kelas = 3

                fully = []
                for i in range(0, kelas):
                    fully.append(np.sum(flatten[image_i] * weight[i, :]) + bias[i])

                ## Exponensial ##
                eksponen = []
                for i in range(0, kelas):
                    eksponen.append(math.exp(fully[i]))

                ## Softmax ##
                softmax = []
                for i in range(0, kelas):
                    softmax.append(eksponen[i] / sum(eksponen))

                ## Cross-entropy Loss ##
                ohl = y[image_i]
                loss = []
                terror = 0.00001

                for i in range(0, kelas):
                    loss.append(ohl[i] * math.log10(softmax[i]))

                loss = sum(loss) * -1

                ## Backpropagation ##

                if loss > terror:
                    cetak = "-> epoch ke" + str(ep + 1)
                    print(cetak)
                    self.textBrowser.append(cetak)
                    delta_y = []
                    for i in range(0, kelas):
                        delta_y.append(softmax[i] - ohl[i])

                    delta_w = np.zeros((kelas, flatten[image_i].shape[0]))
                    for i in range(0, kelas):
                        for j in range(0, flatten[image_i].shape[0]):
                            delta_w[i, j] = delta_y[i] * flatten[image_i][j]

                    delta_b = delta_y

                    ## SGD ##
                    weight_baru = np.zeros_like(weight)
                    for i in range(0, kelas):
                        for j in range(0, weight.shape[1]):
                            weight_baru[i, j] = weight[i, j] - (lrate * delta_w[i, j])

                    weight = weight_baru

                    bias_baru = bias
                    for i in range(0, kelas):
                        bias_baru[i] = bias[i] - (lrate * delta_b[i])
                    bias = bias_baru

                    # FULLY LAGI #
                    fully = []
                    for i in range(0, kelas):
                        fully.append(np.sum(flatten[image_i] * weight[i, :]) + bias[i])

                    ## Exponensial ##
                    eksponen = []
                    for i in range(0, kelas):
                        eksponen.append(math.exp(fully[i]))

                    ## Softmax ##
                    softmax = []
                    for i in range(0, kelas):
                        softmax.append(eksponen[i] / sum(eksponen))

                    cetak = "--> Softmax" + str(softmax)
                    print(cetak)
                    self.textBrowser.append(cetak)

                # np.savez("bobot/filter.npz",f1=l1_filter, f2=l2_filter, f3=l3_filter, f4=l4_filter, f5=l5_filter)

                # lokasi_weight = "bobot/0,000125/bobot" + str(ep + 1) + ".npz"
                #
                # np.savez(lokasi_weight, bobot=weight, biass=bias)

                # if(ep+1 == 50):
                #     np.save("bobot/0,001/ep50/weight.npy", weight)
                #     np.save("bobot/0,001/ep50/bias.npy", bias)
                # elif(ep+1 == 100):
                #     np.save("bobot/0,001/ep100/weight.npy", weight)
                #     np.save("bobot/0,001/ep100/bias.npy", bias)
                # elif(ep+1 == 150):
                #     np.save("bobot/0,001/ep150/weight.npy", weight)
                #     np.save("bobot/0,001/ep150/bias.npy", bias)
                # elif(ep+1 == 200):
                #     np.save("bobot/0,001/ep200/weight.npy", weight)
                #     np.save("bobot/0,001/ep200/bias.npy", bias)
                # elif(ep+1 == 250):
                #     np.save("bobot/0,000125/ep250/weight.npy", weight)
                #     np.save("bobot/0,000125/ep250/bias.npy", bias)

            bar.update(x + 1)
            time.sleep(0.1)
            x=x+1

            bar.finish()

        end = time.time()
        cetak = "Lamanya Proses : " + str(int(round(end - start))) + "s"
        print(cetak)
        self.textBrowser.append(cetak)

        #### END #####

    #### END ###

    def setupUi(self, TrainingCNN):
        TrainingCNN.setObjectName("TrainingCNN")
        TrainingCNN.resize(587, 444)
        self.centralwidget = QtWidgets.QWidget(TrainingCNN)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 110, 571, 311))
        self.groupBox.setObjectName("groupBox")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setGeometry(QtCore.QRect(10, 20, 551, 281))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(240, 50, 111, 51))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        TrainingCNN.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(TrainingCNN)
        self.statusbar.setObjectName("statusbar")
        TrainingCNN.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.trainingCNN)

        self.retranslateUi(TrainingCNN)
        QtCore.QMetaObject.connectSlotsByName(TrainingCNN)

    def retranslateUi(self, TrainingCNN):
        _translate = QtCore.QCoreApplication.translate
        TrainingCNN.setWindowTitle(_translate("TrainingCNN", "Training - Prediksi Kepribadian"))
        self.groupBox.setTitle(_translate("TrainingCNN", "Hasil Softmax Per Iterasi"))
        self.textBrowser.setHtml(_translate("TrainingCNN",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:10.25pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.pushButton.setText(_translate("TrainingCNN", "Training CNN"))
        self.label.setText(_translate("TrainingCNN", "Tahap Training CNN"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    TrainingCNN = QtWidgets.QMainWindow()
    ui = Ui_TrainingCNN()
    ui.setupUi(TrainingCNN)
    TrainingCNN.show()
    sys.exit(app.exec_())

