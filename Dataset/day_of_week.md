# Day_of_week
* This is a file of Description of day_of_the_week cloumn in train and test
## Goal
* Add the feature: day of the week to train and test data.
## Method
* Excel functions:
    1. Concat 
        1. ```arrival_date_year```
        2. ```arrival_date_month```
        3. ```arrival_date_day_of_month```
        * with 
        ```
        =(DATE('arrival_date_year',MONTH(DATEVALUE('arrival_date_month'&"THE REPRESENT MONTH IN NUM")),'arrival_date_day_of_month'))
        ```
        * eg.  
        ```
        =(DATE(2015,MONTH(DATEVALUE("July"&"7")),1))    //returns 2015/7/1
        ```
    2. Get day of the week from the concat result
        * with
        ```
        =WEEKDAY(AD1)-1     //replace AD1 as your position where concate id stored.
        ```
        * You need to change the *Format cell* to *Number* with **NO** number of decimal places
        * [reference link(Microsoft Docs)](https://docs.microsoft.com/en-us/office/troubleshoot/excel/format-cells-settings)