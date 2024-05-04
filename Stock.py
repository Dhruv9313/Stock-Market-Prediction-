from tkinter import *
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.filedialog as fd
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import shutil

# Database connection
con = sqlite3.connect("testdata.db")
cur = con.cursor()


# Function to open the DataFlair File Manager window
def open_file_manager():
    window.withdraw()  # Hide the login window
    
    # Function to adani file
    def adani_file():
        # Load file handling code here...
        np.random.seed(0)
        dates = pd.date_range(start='2017-01-10', end='2017-10-15', freq='B')
        prices = np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        print(df.head())

        # Add some noise to simulate real-world data
        noise = np.random.normal(scale=5, size=len(df))
        df['Price'] += noise
        print(df.head())

        # Read CSV file with 'Date' and 'High' columns
        new_data = pd.read_csv('AdaniPorts.csv', encoding='utf-8')
        print(new_data.columns) 

        # Convert 'Date' column to datetime if not already in that format
        new_data['Date'] = pd.to_datetime(new_data['Date'])

        # Merge dataframes based on 'Date' column
        df = pd.merge(df, new_data[['Date', 'High', 'Close']], on='Date', how='inner')
        print(df.head())

        # Debugging: Print first few rows of merged dataframes
        print(new_data.head())
        print(df.head())

        # Check the datatype of the 'Date' column in both dataframes
        print("Datatype of 'Date' column in new_data:", new_data['Date'].dtype)
        print("Datatype of 'Date' column in df:", df['Date'].dtype)

        # Check if there are common dates between the two dataframes
        common_dates = pd.merge(df, new_data, on='Date', how='inner')
        print("Number of common dates:", len(common_dates))

        # Print some common dates for further inspection
        print(common_dates.head())
       
        # Plot the data
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        sns.lineplot(x='Date', y='Close', data=df, label='Prection', ax=axes[0])  
        sns.lineplot(x='Date', y='High', data=df, label='High', ax=axes[0])  
        axes[0].set_title('Adani Ports Stock Market Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price/High')
        axes[0].legend()

        X = df[['High']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        axes[1].scatter(X_test, y_test, color='blue', label='Actual')
        axes[1].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        axes[1].set_title('Regression Analysis Year 2017')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('High')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('2017-%m-%d'))
        axes[1].legend()

        plt.tight_layout()
        plt.show()



    # Function to apple file
    def apple_file():
        # Load file handling code here...
        np.random.seed(0)
        dates = pd.date_range(start='2017-01-10', end='2017-10-15', freq='B')
        prices = np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        print(df.head())

        # Add some noise to simulate real-world data
        noise = np.random.normal(scale=5, size=len(df))
        df['Price'] += noise
        print(df.head())

        # Read CSV file with 'Date' and 'High' columns
        new_data = pd.read_csv('AppleInc.csv', encoding='utf-8')
        print(new_data.columns) 

        # Convert 'Date' column to datetime if not already in that format
        new_data['Date'] = pd.to_datetime(new_data['Date'])

        # Merge dataframes based on 'Date' column
        df = pd.merge(df, new_data[['Date', 'High', 'Close']], on='Date', how='inner')
        print(df.head())

        # Debugging: Print first few rows of merged dataframes
        print(new_data.head())
        print(df.head())

        # Check the datatype of the 'Date' column in both dataframes
        print("Datatype of 'Date' column in new_data:", new_data['Date'].dtype)
        print("Datatype of 'Date' column in df:", df['Date'].dtype)

        # Check if there are common dates between the two dataframes
        common_dates = pd.merge(df, new_data, on='Date', how='inner')
        print("Number of common dates:", len(common_dates))

        # Print some common dates for further inspection
        print(common_dates.head())

        # Plot the data
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        sns.lineplot(x='Date', y='Close', data=df, label='Prediction', ax=axes[0])  
        sns.lineplot(x='Date', y='High', data=df, label='High', ax=axes[0])  
        axes[0].set_title('Apple Inc Stock Market Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price/High')
        axes[0].legend()

        X = df[['High']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        axes[1].scatter(X_test, y_test, color='blue', label='Actual')
        axes[1].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        axes[1].set_title('Regression Analysis Year 2017')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('High')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('2017-%m-%d'))
        axes[1].legend()

        plt.tight_layout()
        plt.show()


    # Function to tata file
    def tata_file():
       # Load file handling code here...
        np.random.seed(0)
        dates = pd.date_range(start='2017-01-10', end='2017-10-15', freq='B')
        prices = np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        print(df.head())

        # Add some noise to simulate real-world data
        noise = np.random.normal(scale=5, size=len(df))
        df['Price'] += noise
        print(df.head())

        # Read CSV file with 'Date' and 'High' columns
        new_data = pd.read_csv('TataMotors.csv', encoding='utf-8')
        print(new_data.columns) 

        # Convert 'Date' column to datetime if not already in that format
        new_data['Date'] = pd.to_datetime(new_data['Date'])

        # Merge dataframes based on 'Date' column
        df = pd.merge(df, new_data[['Date', 'High', 'Close']], on='Date', how='inner')
        print(df.head())

        # Debugging: Print first few rows of merged dataframes
        print(new_data.head())
        print(df.head())

        # Check the datatype of the 'Date' column in both dataframes
        print("Datatype of 'Date' column in new_data:", new_data['Date'].dtype)
        print("Datatype of 'Date' column in df:", df['Date'].dtype)

        # Check if there are common dates between the two dataframes
        common_dates = pd.merge(df, new_data, on='Date', how='inner')
        print("Number of common dates:", len(common_dates))

        # Print some common dates for further inspection
        print(common_dates.head())

        # Plot the data
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        sns.lineplot(x='Date', y='Close', data=df, label='Prediction', ax=axes[0])  
        sns.lineplot(x='Date', y='High', data=df, label='High', ax=axes[0])  
        axes[0].set_title('Tata Motors Stock Market Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price/High')
        axes[0].legend()

        X = df[['High']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        axes[1].scatter(X_test, y_test, color='blue', label='Actual')
        axes[1].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        axes[1].set_title('Regression Analysis Year 2017')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('High')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('2017-%m-%d'))
        axes[1].legend()

        plt.tight_layout()
        plt.show()


    # Function to tesla file
    def tesla_file():
        # Load file handling code here...
        np.random.seed(0)
        dates = pd.date_range(start='2017-01-10', end='2017-10-15', freq='B')
        prices = np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        print(df.head())

        # Add some noise to simulate real-world data
        noise = np.random.normal(scale=5, size=len(df))
        df['Price'] += noise
        print(df.head())

        # Read CSV file with 'Date' and 'High' columns
        new_data = pd.read_csv('TeslaInc.csv', encoding='utf-8')
        print(new_data.columns) 

        # Convert 'Date' column to datetime if not already in that format
        new_data['Date'] = pd.to_datetime(new_data['Date'])

        # Merge dataframes based on 'Date' column
        df = pd.merge(df, new_data[['Date', 'High', 'Close']], on='Date', how='inner')
        print(df.head())

        # Debugging: Print first few rows of merged dataframes
        print(new_data.head())
        print(df.head())

        # Check the datatype of the 'Date' column in both dataframes
        print("Datatype of 'Date' column in new_data:", new_data['Date'].dtype)
        print("Datatype of 'Date' column in df:", df['Date'].dtype)

        # Check if there are common dates between the two dataframes
        common_dates = pd.merge(df, new_data, on='Date', how='inner')
        print("Number of common dates:", len(common_dates))

        # Print some common dates for further inspection
        print(common_dates.head())

        # Plot the data
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        sns.lineplot(x='Date', y='Close', data=df, label='Prediction', ax=axes[0])  
        sns.lineplot(x='Date', y='High', data=df, label='High', ax=axes[0])  
        axes[0].set_title('Tesla Inc Stock Market Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price/High')
        axes[0].legend()

        X = df[['High']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        axes[1].scatter(X_test, y_test, color='blue', label='Actual')
        axes[1].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        axes[1].set_title('Regression Analysis Year 2017')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('High')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('2017-%m-%d'))
        axes[1].legend()

        plt.tight_layout()
        plt.show()

 # Function to Facebook file
    def Facebook_file():
        # Load file handling code here...
        np.random.seed(0)
        dates = pd.date_range(start='2017-01-10', end='2017-10-17', freq='B')
        prices = np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        print(df.head())

        # Add some noise to simulate real-world data
        noise = np.random.normal(scale=5, size=len(df))
        df['Price'] += noise
        print(df.head())

        # Read CSV file with 'Date' and 'High' columns
        new_data = pd.read_csv('TeslaInc.csv', encoding='utf-8')
        print(new_data.columns) 

        # Convert 'Date' column to datetime if not already in that format
        new_data['Date'] = pd.to_datetime(new_data['Date'])

        # Merge dataframes based on 'Date' column
        df = pd.merge(df, new_data[['Date', 'High', 'Close']], on='Date', how='inner')
        print(df.head())

        # Debugging: Print first few rows of merged dataframes
        print(new_data.head())
        print(df.head())

        # Check the datatype of the 'Date' column in both dataframes
        print("Datatype of 'Date' column in new_data:", new_data['Date'].dtype)
        print("Datatype of 'Date' column in df:", df['Date'].dtype)

        # Check if there are common dates between the two dataframes
        common_dates = pd.merge(df, new_data, on='Date', how='inner')
        print("Number of common dates:", len(common_dates))

        # Print some common dates for further inspection
        print(common_dates.head())

        # Plot the data
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        sns.lineplot(x='Date', y='Close', data=df, label='Prediction', ax=axes[0])  
        sns.lineplot(x='Date', y='High', data=df, label='High', ax=axes[0])  
        axes[0].set_title('Facebook Stock Market Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price/High')
        axes[0].legend()

        X = df[['High']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        axes[1].scatter(X_test, y_test, color='blue', label='Actual')
        axes[1].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        axes[1].set_title('Regression Analysis Year 2017')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('High')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('2017-%m-%d'))
        axes[1].legend()

        plt.tight_layout()
        plt.show()


 # Function to MicrosoftCorp file
    def MicrosoftCorp_file():
        # Load file handling code here...
        np.random.seed(0)
        dates = pd.date_range(start='2017-01-10', end='2017-10-15', freq='B')
        prices = np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        print(df.head())

        # Add some noise to simulate real-world data
        noise = np.random.normal(scale=5, size=len(df))
        df['Price'] += noise
        print(df.head())

        # Read CSV file with 'Date' and 'High' columns
        new_data = pd.read_csv('TeslaInc.csv', encoding='utf-8')
        print(new_data.columns) 

        # Convert 'Date' column to datetime if not already in that format
        new_data['Date'] = pd.to_datetime(new_data['Date'])

        # Merge dataframes based on 'Date' column
        df = pd.merge(df, new_data[['Date', 'High', 'Close']], on='Date', how='inner')
        print(df.head())

        # Debugging: Print first few rows of merged dataframes
        print(new_data.head())
        print(df.head())

        # Check the datatype of the 'Date' column in both dataframes
        print("Datatype of 'Date' column in new_data:", new_data['Date'].dtype)
        print("Datatype of 'Date' column in df:", df['Date'].dtype)

        # Check if there are common dates between the two dataframes
        common_dates = pd.merge(df, new_data, on='Date', how='inner')
        print("Number of common dates:", len(common_dates))

        # Print some common dates for further inspection
        print(common_dates.head())

        # Plot the data
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        sns.lineplot(x='Date', y='Close', data=df, label='Prediction', ax=axes[0])  
        sns.lineplot(x='Date', y='High', data=df, label='High', ax=axes[0])  
        axes[0].set_title('Microsoft Corp Stock Market Data')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price/High')
        axes[0].legend()

        X = df[['High']]
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        axes[1].scatter(X_test, y_test, color='blue', label='Actual')
        axes[1].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        axes[1].set_title('Regression Analysis Year 2017')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('High')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('2017-%m-%d'))
        axes[1].legend()

        plt.tight_layout()
        plt.show()



   # Main application window for File Manager
    root = Tk()
    root.title('Stock Companies')
    root.geometry('450x300')
    root.resizable(0, 0)
    root.configure(bg='#897C78') 


    # GUI components
    Label(root, text='Stock Companies Data List', font=("Comic Sans MS", 15), bg='#FDD4B8').pack(pady=10)

    Button(root, text='Adani Ports', width=20,font=("Helvetica",10,"bold"),fg = '#18120F',bg = '#FDD4B8',command=adani_file).pack(pady=5)
    Button(root, text='Apple Inc', width=20,font=("Helvetica",10,"bold"),fg = '#18120F',bg = '#FDD4B8', command=apple_file).pack(pady=5)
    Button(root, text='Tata Motors', width=20,font=("Helvetica",10,"bold"),fg = '#18120F',bg = '#FDD4B8', command=tata_file).pack(pady=5)
    Button(root, text='Tesla Inc', width=20,font=("Helvetica",10,"bold"),fg = '#18120F',bg = '#FDD4B8',  command=tesla_file).pack(pady=5)
    Button(root, text='Facebook', width=20,font=("Helvetica",10,"bold"),fg = '#18120F',bg = '#FDD4B8',  command=Facebook_file).pack(pady=5)
    Button(root, text='Microsoft Corp', width=20,font=("Helvetica",10,"bold"),fg = '#18120F',bg = '#FDD4B8',  command=MicrosoftCorp_file).pack(pady=5)
    root.mainloop()

# Main login window
window = tk.Tk()
window.geometry("450x300")
window.configure(bg = '#897C78')
window.title("Stock Prediction System")

# the label for Username 
name_label = tk.Label(window, text = "Username :",font=("Helvetica",14,"bold"),fg = '#18120F',bg = '#FDD4B8')
name_label.place(x = 40,y = 60)  
    
# the label for Password  
password_label = tk.Label(window,text = "Password :",font=("Helvetica",14,"bold"),fg = '#18120F',bg = '#FDD4B8')
password_label.place(x = 40,y = 110)  

# Username input box
entry_username = tk.Entry(window,width = 20,font=("Helvetica",14))
entry_username.place(x = 170,y = 60)  

#Password input box   
entry_password = tk.Entry(window,width = 20,font=("Helvetica",14),show = '*')
entry_password.place(x = 170,y = 110)  

#Radio button for choosing 1 of 3 options
radio = tk.IntVar()

def login():
    username = entry_username.get()
    password = entry_password.get()

    # Perform database query to check if username and password match
    cur.execute("SELECT * FROM StudentData WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()

    if user:
        open_file_manager()
    else:
        mb.showerror("Invalid Login", "Username or password is incorrect!")

login_button = tk.Button(window, text="LOG IN", width=20, font=("Helvetica",14,"bold"), bg='#C7A196', activebackground='#FDD4B8', command=login)
login_button.place(x = 108,y = 210)

window.mainloop()


