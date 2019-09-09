# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'training.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import math
# import xlsxwriter
from training import Ui_TrainingCNN
from testing import Ui_Testing

class Ui_Training(object):

    #### Fungsi Open File ####
    def openFileNameDialog(self):
        global img
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "Pilih Tulisan Tangan", "","Image Files (*.JPG)", options=options)
        if fileName:
            img = cv2.imread(fileName)
            cv2.imshow("Tulisan Tangan",img)
            self.btn_Grayscale.setEnabled(True)
            self.tableWidget.setColumnCount(300)
            self.tableWidget.setRowCount(300)
            for x in range(0, 300):
                for y in range(0, 30):
                    nilai_warna = str(img[x,y])
                    self.tableWidget.setItem(x,y,QtWidgets.QTableWidgetItem(nilai_warna))
    #### END ####

    #### Fungsi Prepro Grayscale ####
    def grayscale(self):
        global gray_img
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.destroyWindow("Tulisan Tangan")
        cv2.imshow("Tulisan Tangan Grayscale",gray_img)
        self.btn_Grayscale.setEnabled(False)
        self.btn_Thres.setEnabled(True)

        # self.tableWidget.setColumnCount(0)
        # self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(gray_img.shape[0])
        self.tableWidget.setRowCount(gray_img.shape[1])
        for x in range(0, gray_img.shape[0]):
            for y in range(0, gray_img.shape[1]):
                nilai_warna = str(gray_img[x,y])
                self.tableWidget.setItem(x,y,QtWidgets.QTableWidgetItem(nilai_warna))
    #### END ####

    #### Fungsi Prepro Threshold ####
    def threshold(self):
        global thres_img
        _, thres_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV)
        cv2.destroyWindow("Tulisan Tangan Grayscale")
        cv2.imshow("Tulisan Tangan Threshold",thres_img)
        self.btn_Thres.setEnabled(False)
        self.btn_Segmentasi.setEnabled(True)
        # self.tableWidget.setColumnCount(0)
        # self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(thres_img.shape[0])
        self.tableWidget.setRowCount(thres_img.shape[1])
        for x in range(0, thres_img.shape[0]):
            for y in range(0, thres_img.shape[1]):
                nilai_warna = str(thres_img[x,y])
                self.tableWidget.setItem(x,y,QtWidgets.QTableWidgetItem(nilai_warna))
    #### END ####

    ### Fungsi Prepro Segmentasi ###
    def segmentasi(self):
        global crop_img

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
        cv2.destroyWindow("Tulisan Tangan Threshold")
        cv2.imshow("Tulisan Tangan Crop",crop_img)
        self.btn_Segmentasi.setEnabled(False)
        self.btn_Resize.setEnabled(True)
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(crop_img.shape[0])
        self.tableWidget.setRowCount(crop_img.shape[1])
        # cv2.imwrite('resp0_resize.jpg',crop_img)
        for m in range(0, crop_img.shape[0]):
            for n in range(0, crop_img.shape[1]):
                nilai_warna = str(crop_img[m,n])
                self.tableWidget.setItem(m,n,QtWidgets.QTableWidgetItem(nilai_warna))
    #### END ####

    ### Fungsi Prepro Resize ###
    def resize(self):
        global resize_img

        pa = crop_img.shape[0]
        la = crop_img.shape[1]

        resize_img = np.zeros(shape=[300, 300], dtype=np.uint8)
        for i in range(0, pa-1):
            for j in range(0, la-1):
                if (crop_img[i, j] == 255):
                    x = math.floor((300 * i) / pa)
                    y = math.floor((300 * j) / la)
                    resize_img[x, y] = crop_img[i, j]

        cv2.destroyWindow("Tulisan Tangan Crop")
        cv2.imshow("Tulisan Tangan Resize",resize_img)

        self.btn_Resize.setEnabled(False)
        self.btn_Klasifikasi.setEnabled(True)
        self.tableWidget.setColumnCount(resize_img.shape[0])
        self.tableWidget.setRowCount(resize_img.shape[1])
        for x in range(0, resize_img.shape[0]):
            for y in range(0, resize_img.shape[1]):
                nilai_warna = str(resize_img[x,y])
                self.tableWidget.setItem(x,y,QtWidgets.QTableWidgetItem(nilai_warna))
        # cv2.imwrite('resp0_resize.jpg',resize_img)

        for i in range(0, resize_img.shape[0]):
            for j in range(0, resize_img.shape[1]):
                if(resize_img[i,j]==255):
                    resize_img[i,j]=resize_img[i,j]

        # workbook = xlsxwriter.Workbook('resp.xlsx')
        # worksheet = workbook.add_worksheet()
        # row = 0
        # for col, data in enumerate(resize_img):
        #     worksheet.write_row(col, row, data)
        # workbook.close()

    #### END ####

    def traincnnWindow(self):
        cv2.destroyWindow("Tulisan Tangan Resize")
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_TrainingCNN()
        self.ui.setupUi(self.window)
        self.window.show()
        self.btn_Testing.setEnabled(True)

    def testcnnWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Testing()
        self.ui.setupUi(self.window)
        self.window.show()

    def setupUi(self, Training):
        Training.setObjectName("Training")
        Training.resize(759, 526)
        self.centralwidget = QtWidgets.QWidget(Training)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(610, 30, 131, 261))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btn_Resize = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_Resize.setEnabled(False)
        self.btn_Resize.setGeometry(QtCore.QRect(10, 200, 111, 51))
        self.btn_Resize.setObjectName("btn_Resize")
        self.btn_Resize.clicked.connect(self.resize)

        self.btn_Grayscale = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_Grayscale.setEnabled(False)
        self.btn_Grayscale.setGeometry(QtCore.QRect(10, 20, 111, 51))
        self.btn_Grayscale.setObjectName("btn_Grayscale")
        self.btn_Grayscale.clicked.connect(self.grayscale)

        self.btn_Thres = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_Thres.setEnabled(False)
        self.btn_Thres.setGeometry(QtCore.QRect(10, 80, 111, 51))
        self.btn_Thres.setObjectName("btn_Thres")
        self.btn_Thres.clicked.connect(self.threshold)

        self.btn_Segmentasi = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_Segmentasi.setEnabled(False)
        self.btn_Segmentasi.setGeometry(QtCore.QRect(10, 140, 111, 51))
        self.btn_Segmentasi.setObjectName("btn_Segmentasi")
        self.btn_Segmentasi.clicked.connect(self.segmentasi)

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 130, 591, 351))
        self.groupBox_3.setObjectName("groupBox_3")
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox_3)
        self.tableWidget.setGeometry(QtCore.QRect(10, 20, 571, 321))
        self.tableWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setDefaultSectionSize(30)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(30)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)
        self.tableWidget.verticalHeader().setMinimumSectionSize(30)

        self.groupBox_1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_1.setGeometry(QtCore.QRect(10, 30, 591, 91))
        self.groupBox_1.setObjectName("groupBox_1")
        self.btn_Upload = QtWidgets.QPushButton(self.groupBox_1)
        self.btn_Upload.setGeometry(QtCore.QRect(10, 30, 91, 51))
        self.btn_Upload.setObjectName("btn_Upload")
        self.btn_Upload.clicked.connect(self.openFileNameDialog)

        self.label = QtWidgets.QLabel(self.groupBox_1)
        self.label.setGeometry(QtCore.QRect(110, 10, 91, 16))
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_1)
        self.comboBox.setGeometry(QtCore.QRect(110, 30, 471, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.btn_Kembali = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Kembali.setGeometry(QtCore.QRect(630, 440, 91, 31))
        self.btn_Kembali.setObjectName("btn_Kembali")
        self.btn_Klasifikasi = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Klasifikasi.setEnabled(False)
        self.btn_Klasifikasi.setGeometry(QtCore.QRect(620, 300, 111, 51))
        self.btn_Klasifikasi.setObjectName("btn_Klasifikasi")
        self.btn_Klasifikasi.clicked.connect(self.traincnnWindow)

        self.btn_Testing = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Testing.setEnabled(False)
        self.btn_Testing.setGeometry(QtCore.QRect(620, 360, 111, 51))
        self.btn_Testing.setObjectName("btn_Testing")
        self.btn_Testing.clicked.connect(self.testcnnWindow)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 0, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        Training.setCentralWidget(self.centralwidget)

        self.retranslateUi(Training)
        QtCore.QMetaObject.connectSlotsByName(Training)

    def retranslateUi(self, Training):
        _translate = QtCore.QCoreApplication.translate
        Training.setWindowTitle(_translate("Training", "Preprocessing - Prediksi Kepribadian"))
        self.groupBox_2.setTitle(_translate("Training", "Preprocessing"))
        self.btn_Resize.setText(_translate("Training", "Resize"))
        self.btn_Grayscale.setText(_translate("Training", "Grayscale"))
        self.btn_Thres.setText(_translate("Training", "Threshold"))
        self.btn_Segmentasi.setText(_translate("Training", "Segmentasi"))
        self.groupBox_3.setTitle(_translate("Training", "Matriks"))
        self.groupBox_1.setTitle(_translate("Training", "Tulisan Tangan"))
        self.btn_Upload.setText(_translate("Training", "Pilih \n"
"Tulisan\n"
"Tangan"))
        self.label.setText(_translate("Training", "Kelas Kepribadian"))
        self.comboBox.setItemText(0, _translate("Training", "Optimis"))
        self.comboBox.setItemText(1, _translate("Training", "Pesimis"))
        self.comboBox.setItemText(2, _translate("Training", "Seimbang"))
        self.btn_Kembali.setText(_translate("Training", "Kembali"))
        self.btn_Klasifikasi.setText(_translate("Training", "Proses Training"))
        self.btn_Testing.setText(_translate("Training", "Proses Testing"))
        self.label_2.setText(_translate("Training", "Tahap Preprocessing"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Training = QtWidgets.QMainWindow()
    ui = Ui_Training()
    ui.setupUi(Training)
    Training.show()
    sys.exit(app.exec_())

