# SWEforWebApps-Stocks
Stock trading webapp simulator made for my SWE graduate level course.
Zip file contains the root folder. This includes the Django generated files as well as all of the templates and static images used in the project.

# Python Dependencies
- copy
- django
- datetime
- yfinance
- datetime
- json
- keras
- os
- pandas
- matplotlib
- pyplot
- sklearn (or sci-kit)
- numpy
---------------------------------

# HOW TO RUN THE SERVER/PROGRAM
1) Install dependencies - either global or through Python environment (then activate environment)
2) Change Directory to stocksHW3
3) Start Server using "python manage.py runserver" --> Follow given URL
4) Follow urls.py or just append 'stocksWebApp/about, stocksWebApp/service, etc' to the host url
5) If you want to run the dataPrediction (WARNING - IT WILL OVERWRITE THE PLOTS IF SOMETHING IS CHANGED), can run dataPrediction.py separately
  I reccommend doing this after trying the web-application/simulator at least once.
6) No pop-up dialogs are used so if you want to see console logging, refer to the browser developer tools or the console used to run the server for Python logs.
7) If you want to look at the databse, Visual Studio Code has an extension that presents a good GUI for the db.sqlite3 file 

- IMPORTANT - Use the user, testPort,  or make a new user to be able to use the Transaction/trading system. (refer to testPort password below)

---------------------------------
GENERAL STRUCTURE AND INFORMATION
---------------------------------
After starting/running the server, navigate to '{host:port}/stocksWebApp/about' to get to
the main/about page. Here you can press the buttons (except 'my profile') to navigate to the other pages/views. 'Data Presenting' shows the closing prices of 5 different datasets/tickers
shown across a roughly 90 day/3 month period. 

There is also a Python script named 'yahoo_finance_fetch.py" which shows the process in which
the 'stock_data' JSON file was generated for the previous assignment. For this assignment, the same script generates a different JSON file named 'summerMonths_stock_data.json'. It also shows how the plot used in the Data Presenting view/page is generated. 


-------------------
FINAL PROJECT STUFF -> Can Ignore except for the accounts -> Mainly used as a checklist throughout the development process -> may not be up to date

-------------------

- MILESTONE 1 
  - Used Django's Auth views to create a login system
    - Created a superuser to test login functionality before creating a signup page.
    superuser: test
    pass:swewebapps123

    - Created a login page for existing users
    - Created a registration page for new users
      - acct2: test2
      - pass2: swewebapps1234

- Last Milestones
  - creates portfolio
  - need new users
    - User: testPort
    - Pass: swewebapps1234

- PHASE 2
  - Created Stocks, StockPrice Models for the SQLite3 Database
  - In service.html, have a dropdown so user can select the Stocks
    - When the server is loaded, the stock data from the database gets loaded
  
  - TODO:
    - Form Charts to show price trends when user selects a stock = DONE
    - When user selects a stock, on the same chart show predicted trend of closing prices = DONE
      - Meaning - do the price prediction
        - IDEA - use Light GBM first since it seems to be the fastest -- NEVER IMPLEMENTED IN TIME
          - If done earlier than planned (doubt) - use other Models
          - show report of performance metrics to user
        - LSTM - DONE except stats
          - output plots done with historical data - Done - need to extrapolate into D3 js or just use the output plots
          - need to achieve future predictions - Attempted, not good results 
          - need to do the same with other metrics
          - obtain statistics evaluations (F1, MSE, R-Score, etc)
    - Chart done - JUST FIX COLORS
    
    - When chart is shown, create game mechanics
      - Fake currency -- done by just adding a default balance amount -- DONE
      - Add Transaction model to Database -- DONE
      - Add portfolio model to database -- DONE

    - LAST WEEKEND FINAL STRETCH STEPS
      - Do Transaction system
        - Give user some fake money to buy using the most recent closing price
        - Make sure qty is appropriate with balance, etc.
        - Update portfolio when transaction is done
        - done - sqlite db does not update properly, but the core functionality works
      - GET STATS FOR THE LSTM
        - HAVE A OPTION TO SHOW STATS AND LSTM OUTPUTS
          - IN THIS CONTAINER SHOW THE GENERATED PLOT
          - AND THE LSTM STATISTICS
      - FINALIZE CSS FOR ALL PAGES
      - FINISH REPORT -- DONE
      
    
