# Dataset List
1. train.csv: Raw data of train feature
2. train_label: Raw data of train label
3. test.csv: Raw data of test feature
4. test_nolabel: Raw data of test label
5. train_day_of_week.csv: train.csv + `concat_date` and `day_of_the_week`, which is generated as next part
6. test_day_of_week.csv: test.csv + `concat_date` and `day_of_the_week`, which is generated as next part
7. train_final.csv: Actual train feature generated by preprocessing.py using train_day_of_week.csv
8. test_final.csv: Actual test feature generated by preprocessing.py using test_day_of_week.csv


# How does feature ```day_of_the_week``` be generated?
## Goal
* By the different traveling pattern may be at different day of week, we expect this can help our prediction more precisely.
## Method
We implement by Excel functions:
1. Concat 
    1. `arrival_date_year`
    2. `arrival_date_month`
    3. `arrival_date_day_of_month`
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
