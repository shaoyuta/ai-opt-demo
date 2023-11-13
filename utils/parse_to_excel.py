import re
from xlwt import *

# Workbook is created 
wb = Workbook()
core_sheet = wb.add_sheet('50cores')

# Configuration style
borders = Borders()
borders.left = 1
borders.right = 1
borders.top = 1
borders.bottom = 1

style = XFStyle()
style.borders = borders

ds_log_path='./c_56.log'
row=0
column=1
fre_row=0

#Read output log file
with open(ds_log_path, 'r') as ds_log:
    lines = ds_log.readlines()
    frequencys=("2.0Ghz","2.2Ghz","2.4Ghz","2.6Ghz","2.8Ghz","3.0Ghz","3.2Ghz","3.4Ghz","3.6Ghz","3.8Ghz")
    for line in lines:
        # Get latency
        if re.search("Inference latency:", line):
            latency=re.findall('\d+\.\d+', line)[0]
            print(latency)
            core_sheet.write(row, column, float(latency), style)
            row+=1
        # Get first_token
        if re.search("First token average latency:", line):
            first_token=re.findall('\d+\.\d+', line)[0]
            print(first_token)
            core_sheet.write(row, column, float(first_token), style)
            row+=1
        # Get second_token
        if re.search("Average 2... latency:", line):
            second_token=re.findall('\d+\.\d+', line)[0]
            print(second_token)
            core_sheet.write(row, column, float(second_token), style)
            row+=1
        # Get loop times
        if re.search("Current\s+\{[0-9]\}.*", line):
            loop_time=re.findall('\{([0-9])\}', line)[0]
            current_frequency=re.findall('\d.\dGhz', line)[0]
            for frequency in frequencys:
                if int(loop_time) <= 3 and current_frequency == frequency:
                    column+=1
                    row-=3
                if int(loop_time) == 3 and current_frequency == frequency:
                    row+=3
                    column=1
                    merge_num = int(fre_row) + 2
                    print(fre_row, merge_num)
                    core_sheet.write_merge(fre_row, merge_num , 0, 0, frequency, style)
                    fre_row+=3
# output excel file to current path
wb.save('c_56.xls') 
