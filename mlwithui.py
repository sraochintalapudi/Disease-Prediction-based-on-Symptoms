import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
win=tk.Tk()
win.title("Disease Prediction based on symptoms using Machine Learning")
win.geometry("1400x800")
win.config(background="cyan")
#labeltitle=Label(win,text="Disease Prediction based on symptoms using Machine Learning",font=("Arial",15))
#labeltitle.grid(row=0, column=0)
l1=Label(win,text="Upload Dataset:",font=("Arial",12))
l1.grid(row=0, column=0)
tb2=Text(win, height=25,width=100)
tb2.place(x=21,y=200)
def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",title="Select a File",filetypes=(("CSV files",
                                                      "*.csv"),), )
    data = pd.read_csv("D:/ML Projects/MLproject1/dataset/Training.csv").dropna(axis=1)
    tb2.insert(END,"selected file name is: "+filename+"\n")
    messagebox.showinfo("Message", "Dataset Uploaded successfully")
    tb2.insert(END,"Before Data Preprocessing.....\n")
    tb2.insert(END,data['prognosis'].head())
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])
    tb2.insert(END, "\n After Data Preprocessing ....\n")
    tb2.insert(END, data['prognosis'].head())
    X = data.iloc[:,:-1]
    print(X.shape)
    Y = data.iloc[:, -1]
    print(Y.shape)
    global X_train
    global X_test
    global Y_train
    global Y_test
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.2,random_state = 24)
    tb2.insert(END,f"\nShape of X_train: {X_train.shape}")
    tb2.insert(END,f"\nShape of Y_train: {Y_train.shape}")
    tb2.insert(END,f"\nShape of X_test: {X_test.shape}")
    tb2.insert(END,f"\nShape of Y_test: {Y_test.shape}\n")


label_file_explorer = Label(win,text="Select a File", font=("Arial",12))
tb1=Text(win, height=1,width=40)
tb1.insert(END,"file path")
b1 = Button(win,text="Browse",command=browseFiles,font=("Arial",12))
label_file_explorer.grid(row=1, column=0)
tb1.grid(row=1, column=1)
b1.grid(row=1, column=2)
l2=Label(win,text="Select Algorithm", font=("Arial",12))
l2.grid(row=3, column=0)
var = tk.StringVar()
def print_selection():
    tb2.insert(END,"\nSelected Algorithm is: " + var.get()+"\n")

r1 = tk.Radiobutton(win, text='Support Vector Machine Algorithm', variable=var, value='Support Vector Machine Algorithm', command=print_selection,font=("Arial",12))
r1.grid(row=6, column=0)
r2 = tk.Radiobutton(win, text='Gaussian Naive Bayes Algorithm', variable=var, value='Gaussian Naive Bayes Algorithm', command=print_selection,font=("Arial",12))
r2.grid(row=6, column=1)
r3 = tk.Radiobutton(win, text='Random Forest Algorithm', variable=var, value='Random Forest Algorithm', command=print_selection,font=("Arial",12))
r3.grid(row=6, column=2)
def applyalgorithm():
    if(var.get()=="Support Vector Machine Algorithm"):
        tb2.insert(END,"Algorithm Initiated..........\n")
        svm_model = SVC()
        svm_model.fit(X_train, Y_train)
        svm_preds = svm_model.predict(X_test)
        tb2.insert(END,f"Accuracy using Support Vector Machine Algorithm is: {accuracy_score(Y_test, svm_preds) * 100}\n")
        tb2.insert(END, "Algorithm Completed........\n")
        tb2.insert(END, "Result displayed........\n")
        cf_matrix = confusion_matrix(Y_test, svm_preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for Random Forest")
        plt.show()
    elif(var.get()=="Gaussian Naive Bayes Algorithm"):
        tb2.insert(END, "Algorithm Initiated..........\n")
        nb_model = GaussianNB()
        nb_model.fit(X_train, Y_train)
        nb_preds = nb_model.predict(X_test)
        tb2.insert(END,f"Accuracy using Gaussian Naive Bayes Algorithm is: {accuracy_score(Y_test, nb_preds) * 100}\n")
        tb2.insert(END, "Algorithm Completed........\n")
        tb2.insert(END, "Result displayed........\n")
        cf_matrix = confusion_matrix(Y_test, nb_preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for Random Forest")
        plt.show()
    elif(var.get()=="Random Forest Algorithm"):
        tb2.insert(END, "Algorithm Initiated..........\n")
        rf_model = RandomForestClassifier(random_state=18)
        rf_model.fit(X_train, Y_train)
        rf_preds = rf_model.predict(X_test)
        tb2.insert(END,f"Accuracy using Random Forest Algorithm is: {accuracy_score(Y_test, rf_preds) * 100}\n")
        tb2.insert(END, "Algorithm Completed........\n")
        tb2.insert(END, "Result displayed........\n")
        cf_matrix = confusion_matrix(Y_test, rf_preds)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cf_matrix, annot=True)
        plt.title("Confusion Matrix for Random Forest")
        plt.show()


b2= Button(win,text="Apply Algorithm",command=applyalgorithm,font=("Arial",12))
b2.grid(row=19, column=1)
l3=Label(win,text="OUTPUT:",font=("Arial",12))
l3.grid(row=20, column=0)

win.mainloop()
print("hello")