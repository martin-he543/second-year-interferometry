#!/usr/bin/python
def read_data3(fname):
    ######################################################################
    # A function that reads in the file. Modified from DJC 1/10/2018 by AC 4/03/20
    # Calling arguements:
    #     fname = name of txt file 
    #A better version of read_data2 that extract info out of a text file of any length i with result : [[0], [1], ...... [i]] where [i] is the array containing all the info of the ith column of the txt, all numbers are float, carefull when we have a single column: [ [0] ] array within array !
    
    #3rd - modif to store file names and not floats anymore - used to sort out the export procedure
    ######################################################################
    file = open(fname,"r")#, encoding='mac_roman')
    
    lines= file.readlines()
    signal=[[]]
    i=0
    j=0
    for k in lines[0]:
        #print (k)
        if k == ' ':
            signal.append([])
    #print(signal)
    for line in range(0,len(lines)):
        #print(line,'line')
        j=0
        i=0
        while j<=len(lines[line]) and i<=len(lines[line])-2:   #careful the 2 has been changed here, it was a 1 before !
            dd='' 
            #print(i,j)
            #print(signal)
            while lines[line][i]!=' ' and i<=len(lines[line])-2:
                dd=dd+lines[line][i]
                i=i+1
            
            #print(j)
            #print(dd)
            signal[j].append(float(dd))
            j=j+1
            i=i+1
            
            
    file.close()
    return signal



#file='Result_simu_26_02_f50_V0.26_fsamp9_Tg_a.txt_t0=0_l150_j50.txt'

#file='27_02_f80_g82_h_00_B3_C3D00_fsamp5_Tg_a.txt'

#sig=read_data3(file)
#print(sig)
