
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import itertools as it
import time
pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


loopdG=pd.DataFrame({'size':[3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30],'value':[3.5, 3.5, 3.3, 4.0, 4.2, 4.3, 4.5, 4.6, 5.0, 5.1, 5.3, 5.5, 5.7, 6.1, 6.3]},index=range(1,16),columns=['size', 'value'])
SantaLuciaNN=pd.DataFrame({'Interaction': ['AA', 'AT', 'TA', 'CA', 'GT', 'CT', 'GA', 'CG', 'GC', 'GG'],'deltaH': [-7.9, -7.2, -7.2, -8.5, -8.4, -7.8, -8.2, -10.6, -9.8, -8.0],'deltaS':[-22.2, -20.4, -21.3, -22.7, -22.4, -21.0,-22.2, -27.2, -24.4, -19.9],'deltaG':[-1.0, -0.88, -0.58, -1.45, -1.44, -1.28, -1.3, -2.17, -2.24, -1.84]},index=range(1,11),columns=['Interaction', 'deltaH', 'deltaS', 'deltaG'])
termisdG=pd.DataFrame({'A':[-0.51, -0.96, -0.58, -0.5, -0.12, -0.82, -0.92, -0.48],'C':[-0.42, -0.52, -0.34, -0.02, 0.28, -0.31, -0.23, -0.19],'G':[-0.62, -0.72, -0.56, 0.48, -0.01, -0.01, -0.44, -0.5],'T':[-0.71, -0.58, -0.61, -0.1, 0.13, -0.52, -0.35, -0.29]},index=['XA/T', 'XC/G', 'XG/C', 'XT/A', 'AX/T', 'CX/G', 'GX/C', 'TX/A'],columns=['A', 'C', 'G', 'T'])
termisdH=pd.DataFrame({'A':[0.2, -6.3, -3.7, -2.9, -0.5, -5.9, -2.1, -0.7],'C':[0.6, -4.4, -4.0, -4.1, 4.7, -2.6, -0.2, 4.4],'G':[-1.1, -5.1, -3.9, -4.2, -4.1, -3.2, -3.9, 1.6],'T':[-6.9, -4.0, -4.9, -0.2, -3.8, -5.2, -4.4, 2.9]},index=['XA/T', 'XC/G', 'XG/C', 'XT/A', 'AX/T', 'CX/G', 'GX/C', 'TX/A'],columns=['A', 'C', 'G', 'T'])
tetraloops=pd.DataFrame({'dG':[0.7, 0.2, 0.5, -1.3, -1.6, -1.6, -2.0, -2.1, -1.6, -0.3, -1.6, -1.6, 0.3, -2.1, -1.6, 0.3, -0.7, -0.5, -1.0, 0.9, 0.7, 1.0, 0.0, -0.8, -1.1, -1.1, -1.5, -1.6, -1.1, 0.2, -1.1, -1.0, 0.8, -1.6, -1.1, 0.8, -0.2, 0.0, -0.5, 1.5, 0.7, 1.0, -0.8, -1.1, -1.1, -1.6, -1.6, -1.1, 0.2, -1.1, -1.1, 0.8, -1.6, -1.1, 0.8, -0.2, 0.0, -0.5, -1.5, 1.0, 1.0, -0.5, -1.1, -1.1, -1.6, -1.6, -1.1, -0.1, -1.1, -1.1, 0.8, -1.6, -1.1, -0.5, -0.4, -0.4, -0.5, 0.4, 0.2, 0.5, -1.3, -1.6, -1.6, -2.1, -2.1, -1.6, -0.3, -1.6, -1.6, 0.3, -2.1, -1.6, 0.3, -0.7, -0.5, -1.0, 1.0, 0.5, 0.5, -1.0, -1.5, -1.5, -2.0, -2.0, -1.5, -0.6, -1.5, -1.5, 0.3, -2.0, -1.5, -0.9, -1.5, -0.9, -1.0],'dH':[0.5, 0.7, 1.0, 0.0, -1.1, -1.1, -1.5, -1.6, -1.1, 0.2, -1.1, -1.1, 0.5, -1.6, -1.1, 0.8, -0.2, 0.0, -0.5, 0.5, 0.7, 1.0, 0.0, 0.0, -1.1, -1.1, -1.5, -1.6, -1.1, 0.2, -1.1, -1.0, 0.5, -1.6, -1.1, 0.8, -0.2, 0.0, -0.5, 0.5, 0.7, 1.0, 0.0, -1.1, -1.1, -1.6, -1.6, -1.1, 0.2, -1.1, -1.1, 0.5, -1.6, -1.1, 0.8, -0.2, 0.0, -0.5, 0.5, 1.0, 1.0, 0.0, -1.1, -1.1, -1.6, -1.6, -1.1, -0.1, -1.1, -1.1, 0.5, -1.6, -1.1, -0.5, -0.4, -0.4, -0.5, 0.5, 0.7, 1.0, 0.0, -1.1, -1.1, -1.6, -1.6, -1.1, 0.2, -1.1, -1.1, 0.5, -1.6, -1.1, 0.8, -0.2, 0.0, -0.5, 0.5, 1.0, 1.0, 0.0, -1.0, -1.0, -1.5, -1.5, -1.0, -0.1, -1.0, -1.0, 0.5, -1.5, -1.0, -0.4, -1.0, -0.4, -0.5]},index=['AAAAAT', 'AAAACT', 'AAACAT', 'ACTTGT', 'AGAAAT', 'AGAGAT', 'AGATAT', 'AGCAAT', 'AGCGAT', 'AGCTTT', 'AGGAAT', 'AGGGAT', 'AGGGGT', 'AGTAAT', 'AGTGAT', 'AGTTCT', 'ATTCGT', 'ATTTGT', 'ATTTTT', 'CAAAAG', 'CAAACG', 'CAACAG', 'CAACCG', 'CCTTGG', 'CGAAAG', 'CGAGAG', 'CGATAG', 'CGCAAG', 'CGCGAG', 'CGCTTG', 'CGGAAG', 'CGGGAG', 'CGGGGG', 'CGTAAG', 'CGTGAG', 'CGTTCG', 'CTTCGG', 'CTTTGG', 'CTTTTG', 'GAAAAC', 'GAAACC', 'GAACAC', 'GCTTGC', 'GGAAAC', 'GGAGAC', 'GGATAC', 'GGCAAC', 'GGCGAC', 'GGCTTC', 'GGGAAC', 'GGGGAC', 'GGGGGC', 'GGTAAC', 'GGTGAC', 'GGTTCC', 'GTTCGC', 'GTTTGC', 'GTTTTC', 'GAAAAT', 'GAAACT', 'GAACAT', 'GCTTGT', 'GGAAAT', 'GGAGAT', 'GGATAT', 'GGCAAT', 'GGCGAT', 'GGCTTT', 'GGGAAT', 'GGGGAT', 'GGGGGT', 'GGTAAT', 'GGTGAT', 'GTATAT', 'GTTCGT', 'GTTTGT', 'GTTTTT', 'TAAAAA', 'TAAACA', 'TAACAA', 'TCTTGA', 'TGAAAA', 'TGAGAA', 'TGATAA', 'TGCAAA', 'TGCGAA', 'TGCTTA', 'TGGAAA', 'TGGGAA', 'TGGGGA', 'TGTAAA', 'TGTGAA', 'TGTTCA', 'TTTCGA', 'TTTTGA', 'TTTTTA', 'TAAAAG', 'TAAACG', 'TAACAG', 'TCTTGG', 'TGAAAG', 'TGAGAG', 'TGATAG', 'TGCAAG', 'TGCGAG', 'TGCTTG', 'TGGAAG', 'TGGGAG', 'TGGGGG', 'TGTAAG', 'TGTGAG', 'TTTCGG', 'TTTTAG', 'TTTTGG', 'TTTTTG'],columns=['dG', 'dH'])
triloops=pd.DataFrame({'dG':[-1.5, -1.5, -1.5, -1.5, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.5, -1.5, -1.5, -1.5],'dH':[-1.5, -1.5, -1.5, -1.5, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.5, -1.5, -1.5, -1.5]},index=['AGAAT', 'AGCAT', 'AGGAT', 'AGTAT', 'CGAAG', 'CGCAG', 'CGGAG', 'CGTAG', 'GGAAC', 'GGCAC', 'GGGAC', 'GGTAC', 'TGAAA', 'TGCAA', 'TGGAA', 'TGTAA'],columns=['dG', 'dH'])


# In[3]:


def findindex(self,i):
    try:
        return (self.index(i))
    except:
        return -1
def GCcalc(sequence):
    GCcontent=(sequence.count('C')+sequence.count('G'))/(len(sequence))*100
    return round(GCcontent,1)
def reverseComplement(sequence):
    complement={'A':'T','C':'G','G':'C','T':'A'}
    compseqlist=[complement[sequence[i]] for i in range(len(sequence))]
    rcseq=''.join(compseqlist)
    rcseq=rcseq[::-1]
    return rcseq
def SantaLuciacalc(sequence,Na,Mg,dNTP,oligo):
    Hvalues=0
    Svalues=0
    Gvalues=0
    nncount=0
    for nn in range(0,(len(sequence)-1)):
        nncount=nncount+1
        x=findindex(list(SantaLuciaNN.loc[:,'Interaction']),sequence[nn:nn+2])#The python list start at 0, unlike R [1:2] function, python will only output [1] instead of [1and2]
        if x!=-1:
            Hvalues=Hvalues+SantaLuciaNN.iloc[x,1]
            Svalues=Svalues+SantaLuciaNN.iloc[x,2]
            Gvalues=Gvalues+SantaLuciaNN.iloc[x,3]
        else:
            y=findindex(list(SantaLuciaNN.loc[:,'Interaction']),reverseComplement(sequence[nn:nn+2]))
            Hvalues=Hvalues+SantaLuciaNN.iloc[y,1]
            Svalues=Svalues+SantaLuciaNN.iloc[y,2]
            Gvalues=Gvalues+SantaLuciaNN.iloc[y,3]
    initdH={'G':0.1,'C':0.1,'A':2.3,'T':2.3}
    initdS={'G':-2.8,'C':-2.8,'A':4.1,'T':4.1}
    initdG={'G':0.98,'C':0.98,'A':1.03,'T':1.03}
    Gvalues=Gvalues+initdG[sequence[0]]+initdG[sequence[-1]]+0.43
    Hvalues=Hvalues+initdH[sequence[0]]+initdH[sequence[-1]]+0
    Svalues=Svalues+initdS[sequence[0]]+initdS[sequence[-1]]+(0.368*nncount*np.log(Na+120*(Mg-dNTP)**0.5))
    Tm=((Hvalues*1000)/(Svalues+(1.987*(np.log((oligo)/4)))))-273.15
    return {'Tm':round(Tm,2),'dGvalue':round(Gvalues,1),'dHvalue':round(Hvalues,1),'dSvalue':round(Svalues,1)}
def singlebasestretch(sequence):
    counter=1
    longest=0
    for i in range(0,len(sequence)-1):
            if sequence[i]==sequence[i+1]:
                counter=counter+1
                if counter>longest:
                    longest=counter
            else:
                counter=1
    return longest    
def GCclamp(sequence):
    lastbases=sequence[-4:]
    return lastbases.count('G')+lastbases.count('C')
def DNAString(seq):
    myseq=''
    for i in range(len(seq)):
         myseq=myseq+seq[i]
    return myseq
def fastatextprocessing(seq):
    seq=seq.split('\n')
    seq=[i for i in seq if i!='']#take away any "empty" lines
    nameindex=[i for i in range(0,len(seq)) if '>' in seq[i]]
    return ({'name':nameindex,'sequencelist':seq})
def combinationlist(seqF,seqR):
    seqF=fastatextprocessing(seqF)
    seqR=fastatextprocessing(seqR)
    listFname=[seqF['sequencelist'][i].replace('>','') for i in seqF['name']]
    listRname=[seqR['sequencelist'][i].replace('>','') for i in seqR['name']]
    listFseq=[seqF['sequencelist'][i+1] for i in seqF['name']]
    listRseq=[seqR['sequencelist'][i+1] for i in seqR['name']]
    return ({'namecombo':list(it.product(listFname,listRname)),'seqcombo':list(it.product(listFseq,listRseq))})
def file_save(defaultextension):
    f = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=defaultextension)
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    else:
        f=str(f)
        f=f.split("'")[1]# `()` was missing.
        return(f)

def browsefile():
    f = tkinter.filedialog.askopenfilename()
    if f is None:
        return
    else:
        return (f)


# In[4]:


'''Secondary structures calculation'''
def Tmhairpincalc(dlist3,Na,Mg,dNTP,oligo):
    if not dlist3[0]:
        hpdG=0
        hpdH=0
        hpdS=0
        hpTm=0
    else:
    #check the termisdG and termisdH table.
        termismatch=dlist3[1]
        row1='X'+termismatch[0]+'/'+termismatch[3]
        col1=termismatch[1]
        row2=termismatch[0]+'X'+'/'+termismatch[3]
        col2=termismatch[2]
        termisdGvalue=0
        termisdHvalue=0
        termisdGvalue=termisdG.loc[row1,col1]+termisdG.loc[row2,col2]
        termisdHvalue=termisdH.loc[row1,col1]+termisdH.loc[row2,col2]#The table has to be read from 5' to 3'
        #check loop size
        hpdG=0
        hpdH=0
        loopsize=len(dlist3[0])
        if loopsize==3:
            triloop=dlist3[0]#For the table, the loopstart and loopend must be included
            hpdG=loopdG.iloc[0,1]
            hpdH=0
            if triloop in list(triloops.index): #if the loop is not in the table, the dG value should be only
                #from the loop penalty table
                triloopbonusdG=triloops.loc[triloop,'dG']
                triloopbonusdH=triloops.loc[triloop,'dH']
                hpdG=hpdG+triloopbonusdG
                hpdH=hpdH+triloopbonusdH
            #if the loop is closed by AT apply penalty. read from 5' to 3'
            if triloop[0]=='A' and triloop[-1]=='T':
                hpdG=hpdG+0.5
                hpdH=hpdH+2.2
        if loopsize==4:
            tetraloop=dlist3[0]#For the table, the loopstart and loopend must be included
            hpdG=loopdG.iloc[1,1]+termisdGvalue
            hpdH=termisdHvalue
            if tetraloop in list(tetraloops.index):
                tetraloopbonusdG=tetraloops.loc[tetraloop,'dG']
                tetraloopbonusdH=tetraloops.loc[tetraloop,'dH']
                hpdG=hpdG+tetraloopbonusdG
                hpdH=tetraloopbonusdH
                #for loopsize over 4
        if loopsize>4:
            if loopsize in list(loopdG.loc[:,'size']):
                ndGvalue=loopdG.iloc[list(loopdG.loc[:,'size']).index(loopsize),1]#value in the loopdGtable.
            else:
                ndGvalue=6.3+2.44*(1.9872/1000)*310.15*np.log(loopsize/30)#the formula taken by SantaLucia
                #paper, 30 is the largest size known experimental value
            hpdG=ndGvalue+termisdGvalue
            hpdH=termisdHvalue
        #combine the hairpin loop calculation with the helix stack calculation
        #looppos gives the index of the longest stretch in first half of dlist3
        #then take primer[dlist3] to get the longest stretch
        #combine the hairpin loop and SantaLucia NN hairpin G and H value
        hpdG=round(hpdG+SantaLuciacalc(dlist3[2],Na,Mg,dNTP,oligo)['dGvalue'],1)
        hpdH=round(hpdH+SantaLuciacalc(dlist3[2],Na,Mg,dNTP,oligo)['dHvalue'],1)
        hpdS=round(((hpdH-hpdG)/(273.15+37))*1000,1)
        try:
            hpTm=round((hpdH*1000/hpdS)-273.15,1)
        except ZeroDivisionError:
            hpTm=0
    return hpTm
def findloop(dlist3,primer):
    if not dlist3 or dlist3[-1]-dlist3[0]-1<3:
        loop=[]
        termismatch=[]
        longestrun=[]
    else:
        dlist3half=dlist3[0:int(len(dlist3)/2)]#take into account only the first half.
        dlist3halfcumsum=list(np.diff(dlist3half)-1)
        dlist3halfcumsum.insert(0,1)
        dlist3halfcumsum=list(np.cumsum(dlist3halfcumsum))
        consec=list(set(dlist3halfcumsum))
        rle=[dlist3halfcumsum.count(i) for i in consec]#it will count and compute the longest stretch of consecutive, this is used for Tm calc
        value=consec[findindex(rle,max(rle))]
        looppos=[i for i in range(0,len(dlist3halfcumsum)) if dlist3halfcumsum[i]==value]#find the index of the longest stretch in cumsum list.
        loopstart=dlist3[looppos[-1]]#record the position of the innermost bonds to know where the loop start/end
        loopend=dlist3[len(dlist3)-1-findindex(dlist3,loopstart)]#record the posiition of the innermost bond by looking
        #at find the index of the loopstart within the dlist3, and call the first index in the second half.
        loop=primer[loopstart+1:loopend]# the loop :note that it doesn't include the letter with index of loopstart or loopend
        #loopsize=len(loop)
        longestrun=''.join([primer[i] for i in [dlist3[j] for j in looppos]])
        termismatch=''.join([primer[loopstart:loopstart+2],primer[loopend-1:loopend+1]])
    return [loop,termismatch,longestrun]
def changingmidpoints(dlist3):
    middlepoint1=dlist3[int(len(dlist3)/2-1)]
    middlepoint2=dlist3[int(len(dlist3)/2)]
    if (middlepoint2-middlepoint1)==1 or (middlepoint2-middlepoint1) ==2:
    #if the bases inside the loop is either 1 or 2, take away the innermost bonds to make a loop>=3
        dlist3=[dlist3[i] for i in range(0,len(dlist3)) if (dlist3[i]!=middlepoint1)and(dlist3[i]!=middlepoint2)]
    return dlist3
def secondarystructurecalcSelfdimerHairpinopt(primer,averageta,Na,Mg,dNTP,oligo):
    Shiftby=list(range(-len(primer),len(primer)+1))
    dlist2=list(map(lambda h: [i-h for i,x in enumerate(primer[::-1]) if x==(reverseComplement(primer[h]))],range(len(primer))))
    dlist2biglist=[x for sublist in dlist2 for x in sublist]
    Shiftbylist=list(filter(lambda shiftby: shiftby in dlist2biglist,Shiftby))
    Frequencylist=[dlist2biglist.count(i) for i in Shiftbylist]#this step actually is not necessary for calculating primers
    if not Frequencylist:
    ##do something see below
        '''Frequencylist=[0]*len(Shiftbylist)
        dGprimlist=[0]*len(Shiftbylist)
        hpdG=[0]*len(Shiftbylist)
        hpdH=[0]*len(Shiftbylist)
        hpdS=[0]*len(Shiftbylist)
        hpTm=[0]*len(Shiftbylist)'''
        dGSelfdimer_mean=0
        WorstTmhairpin=0
    else:
        '''self-priming'''
        dlist3=list(map(lambda z: [i for i in range(len(dlist2)) if z in dlist2[i]],Shiftbylist))
        seqlist=list(map(lambda x: [primer[o] for o in x],dlist3))
        seqlist=[''.join(i) for i in seqlist]
        dGvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['dGvalue'] for i in seqlist]
        Tmvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['Tm'] for i in seqlist]
        dlist3=list(map(changingmidpoints,dlist3))#changed midpoints for loops less than 3
        dlist3=list(map(findloop,dlist3,it.repeat(primer)))
        dlist3=list(map(Tmhairpincalc,dlist3,it.repeat(Na),it.repeat(Mg),it.repeat(dNTP),it.repeat(oligo)))
        dGSelfdimer_mean=round(np.mean([i for i in dGvaluelist if i!=0]),2)
        WorstTmhairpin=dlist3[(sorted([(x,abs(i-averageta)) for x,i in enumerate(dlist3)],key=lambda tup:tup[1]))[0][0]]
    return ({'dGSelfdimer_mean':dGSelfdimer_mean,'WorstTmHairpin':WorstTmhairpin})


# In[5]:


'''Drawing and summary table for Self dimer and Hairpin'''
def drawingboxes(box1,box3,box2,drawinglist,primer):
    for n in range(len(drawinglist[1])):
        box2=box2[:drawinglist[1][n]]+'|'+box2[(drawinglist[1][n]+1):]
    move=' '*abs(drawinglist[0])
    if drawinglist[0]<0:
        box1=primer
        box3=move+primer[::-1]
    if drawinglist[0]>0:
        box1=move+primer
        box3=primer[::-1]
        box2=move+box2
    if drawinglist[0]==0:
        box1=primer
        box3=primer[::-1]
    return ((box1,box2,box3),drawinglist[0],drawinglist[2],drawinglist[3])
def findloopanddrawboxes(dlist3,primer):
    if not dlist3 or dlist3[-1]-dlist3[0]-1<3:
        loop=[]
        termismatch=[]
        longestrun=[]
        box1=''
        box2=''
        box3=''
    else:
        dlist3half=dlist3[0:int(len(dlist3)/2)]#take into account only the first half.
        dlist3halfcumsum=list(np.diff(dlist3half)-1)
        dlist3halfcumsum.insert(0,1)
        dlist3halfcumsum=list(np.cumsum(dlist3halfcumsum))
        consec=list(set(dlist3halfcumsum))
        rle=[dlist3halfcumsum.count(i) for i in consec]#it will count and compute the longest stretch of consecutive, this is used for Tm calc
        value=consec[findindex(rle,max(rle))]
        looppos=[i for i in range(0,len(dlist3halfcumsum)) if dlist3halfcumsum[i]==value]#find the index of the longest stretch in cumsum list.
        loopstart=dlist3[looppos[-1]]#record the position of the innermost bonds to know where the loop start/end
        loopend=dlist3[len(dlist3)-1-findindex(dlist3,loopstart)]#record the posiition of the innermost bond by looking
        #at find the index of the loopstart within the dlist3, and call the first index in the second half.
        loop=primer[loopstart+1:loopend]# the loop :note that it doesn't include the letter with index of loopstart or loopend
        loopsize=len(loop)
        box1=(primer[0:loopstart+1]+loop[0:int(len(loop)/2)])[::-1]#first part until the beginning of the loop, which itself has been divided in two part
        box3=loop[math.ceil(len(loop)/2):]+primer[loopend:]#the second part of the loop+ the rest of the primer
        box2=' '*len(box3)
        for n in dlist3[int(len(dlist3)/2):]:#only need to look at the second half of dlist3
            posinsecondhalf=n-(len(primer)-len(box3))#take value in second half (it will be higher value)
            #and substract it to the length of the box1+loop, you will get the index value of the line in box2
            box2=box2[0:posinsecondhalf]+'|'+box2[posinsecondhalf+1:]
        if loopsize%2!=0:
            middlepartloop=(loop.replace(loop[0:int(len(loop)/2)],'',1)).replace(loop[math.ceil(len(loop)/2):],'',1)#replace first with the loop present in box1, and then loop part in box3
            box2=middlepartloop+box2[1:]
        longestrun=''.join([primer[i] for i in [dlist3[j] for j in looppos]])
        termismatch=''.join([primer[loopstart:loopstart+2],primer[loopend-1:loopend+1]])
    return [loop,termismatch,longestrun,(box1,box2,box3)]
def Tmhairpincalcanddraw(dlist3,Na,Mg,dNTP,oligo):
    if not dlist3[0]:
        hpdG=0
        hpdH=0
        hpdS=0
        hpTm=0
    else:
    #check the termisdG and termisdH table.
        termismatch=dlist3[1]
        row1='X'+termismatch[0]+'/'+termismatch[3]
        col1=termismatch[1]
        row2=termismatch[0]+'X'+'/'+termismatch[3]
        col2=termismatch[2]
        termisdGvalue=0
        termisdHvalue=0
        termisdGvalue=termisdG.loc[row1,col1]+termisdG.loc[row2,col2]
        termisdHvalue=termisdH.loc[row1,col1]+termisdH.loc[row2,col2]#The table has to be read from 5' to 3'
        #check loop size
        hpdG=0
        hpdH=0
        loopsize=len(dlist3[0])
        if loopsize==3:
            triloop=dlist3[0]#For the table, the loopstart and loopend must be included
            hpdG=loopdG.iloc[0,1]
            hpdH=0
            if triloop in list(triloops.index): #if the loop is not in the table, the dG value should be only
                #from the loop penalty table
                triloopbonusdG=triloops.loc[triloop,'dG']
                triloopbonusdH=triloops.loc[triloop,'dH']
                hpdG=hpdG+triloopbonusdG
                hpdH=hpdH+triloopbonusdH
            #if the loop is closed by AT apply penalty. read from 5' to 3'
            if triloop[0]=='A' and triloop[-1]=='T':
                hpdG=hpdG+0.5
                hpdH=hpdH+2.2
        if loopsize==4:
            tetraloop=dlist3[0]#For the table, the loopstart and loopend must be included
            hpdG=loopdG.iloc[1,1]+termisdGvalue
            hpdH=termisdHvalue
            if tetraloop in list(tetraloops.index):
                tetraloopbonusdG=tetraloops.loc[tetraloop,'dG']
                tetraloopbonusdH=tetraloops.loc[tetraloop,'dH']
                hpdG=hpdG+tetraloopbonusdG
                hpdH=tetraloopbonusdH
                #for loopsize over 4
        if loopsize>4:
            if loopsize in list(loopdG.loc[:,'size']):
                ndGvalue=loopdG.iloc[list(loopdG.loc[:,'size']).index(loopsize),1]#value in the loopdGtable.
            else:
                ndGvalue=6.3+2.44*(1.9872/1000)*310.15*np.log(loopsize/30)#the formula taken by SantaLucia
                #paper, 30 is the largest size known experimental value
            hpdG=ndGvalue+termisdGvalue
            hpdH=termisdHvalue
        #combine the hairpin loop calculation with the helix stack calculation
        #looppos gives the index of the longest stretch in first half of dlist3
        #then take primer[dlist3] to get the longest stretch
        #combine the hairpin loop and SantaLucia NN hairpin G and H value
        SantaLucialongestrun=SantaLuciacalc(dlist3[2],Na,Mg,dNTP,oligo)
        hpdG=round(hpdG+SantaLucialongestrun['dGvalue'],1)
        hpdH=round(hpdH+SantaLucialongestrun['dHvalue'],1)
        hpdS=round(((hpdH-hpdG)/(273.15+37))*1000,1)
        try:
            hpTm=round((hpdH*1000/hpdS)-273.15,1)
        except ZeroDivisionError:
            hpTm=0
    return [hpdG,hpdH,hpdS,hpTm,dlist3[3]]
def DrawSingleSecStructureSelfdimerHairpinopt(primer,seqname,file,logfile,Na,Mg,dNTP,oligo):
    file.write('###############{} SELF-DIMER#############'.format(seqname))
    print('###############{} SELF-DIMER#############'.format(seqname))
    Shiftby=list(range(-len(primer),len(primer)+1))
    dlist2=list(map(lambda h: [i-h for i,x in enumerate(primer[::-1]) if x==(reverseComplement(primer[h]))],range(len(primer))))
    dlist2biglist=[x for sublist in dlist2 for x in sublist]
    Shiftbylist=list(filter(lambda shiftby: shiftby in dlist2biglist,Shiftby))
    Frequencylist=[dlist2biglist.count(i) for i in Shiftbylist]#this step actually is not necessary for calculating primers
    if not Frequencylist:
    ##do something see below
        Frequencylist=[0]*len(Shiftbylist)
        dGprimlist=[0]*len(Shiftbylist)
        hpdG=[0]*len(Shiftbylist)
        hpdH=[0]*len(Shiftbylist)
        hpdS=[0]*len(Shiftbylist)
        hpTm=[0]*len(Shiftbylist)
        dGSelfdimer_mean=0
        WorstTmhairpin=0
    else:
        '''self-priming'''
        dlist3=list(map(lambda z: [i for i in range(len(dlist2)) if z in dlist2[i]],Shiftbylist))
        seqlist=list(map(lambda x: [primer[o] for o in x],dlist3))
        seqlist=[''.join(i) for i in seqlist]
        dGvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['dGvalue'] for i in seqlist]
        dGprimlist=dGvaluelist
        Tmvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['Tm'] for i in seqlist]
        '''drawing'''
        box2=[' '*len(primer) for i in Shiftbylist]
        box1=box3=['' for i in Shiftbylist]
        drawinglist=list(zip(Shiftbylist,dlist3,dGvaluelist,Tmvaluelist))
        draw=list(map(drawingboxes,box1,box3,box2,drawinglist,it.repeat(primer)))
        #print boxes
        for i in range(len(draw)):
            print('\n'.join((draw[i][0])))
            print('It has shifted by {0} position, and this structure has delta G value of {1}, and Tm value of {2}'.format(draw[i][1],draw[i][2],draw[i][3]))
            file.write('\n')
            file.write('\n'.join((draw[i][0])))
            file.write('It has shifted by {0} position, and this structure has delta G value of {1}, and Tm value of {2}'.format(draw[i][1],draw[i][2],draw[i][3]))
        dlist3=list(map(changingmidpoints,dlist3))#changed midpoints for loops less than 3
        dlist3=list(map(findloopanddrawboxes,dlist3,it.repeat(primer)))
        drawinglisthp=list(map(Tmhairpincalcanddraw,dlist3,it.repeat(Na),it.repeat(Mg),it.repeat(dNTP),it.repeat(oligo)))
        print('\n')
        print('###############{} HAIRPIN############'.format(seqname))
        print('\n')
        file.write('\n')
        file.write('###############{} HAIRPIN############'.format(seqname))
        file.write('\n')
        for i in range(len(drawinglisthp)):
            if not drawinglisthp[i][4][1]:
                continue
            print('\n'.join((drawinglisthp[i][4])))
            print('deltaG is {0}, deltaH is {1}, deltaS is {2}, Tm is {3}'.format(drawinglisthp[i][0],
                                                                                  drawinglisthp[i][1],
                                                                                  drawinglisthp[i][2],
                                                                                  drawinglisthp[i][3]))
            print('\n')
            file.write('\n'.join((drawinglisthp[i][4])))
            file.write('\n')
            file.write('deltaG is {0}, deltaH is {1}, deltaS is {2}, Tm is {3}'.format(drawinglisthp[i][0],
                                                                                  drawinglisthp[i][1],
                                                                                  drawinglisthp[i][2],
                                                                                  drawinglisthp[i][3]))
            file.write('\n')
        number3=pd.DataFrame(data={'Shiftby':Shiftbylist,'Frequencylist': Frequencylist, 'dGprim': dGprimlist,'Tmprim':Tmvaluelist,
                      'dGhairpin':[a[0] for a in drawinglisthp],
                      'dHhairpin':[a[1] for a in drawinglisthp],
                      'dShairpin':[a[2] for a in drawinglisthp],
                      'TmHairpin':[a[3] for a in drawinglisthp]},columns=['Shiftby','Frequencylist','dGprim','Tmprim','dGhairpin','dHhairpin','dShairpin','TmHairpin'])
        number3.to_excel(logfile,sheet_name='{}'.format(seqname),index=False,freeze_panes=[1,1])


# In[6]:


'''Calculating Primer dimer'''
def PDcalc(primer1,primer2,Na,Mg,dNTP,oligo):
    Shiftby=list(range(-len(primer1),len(primer1)+1))
    dlist2=list(map(lambda h: [i-h for i,x in enumerate(primer2) if x==(reverseComplement(primer1[h]))],range(len(primer1))))
    dlist2biglist=[x for sublist in dlist2 for x in sublist]
    Shiftbylist=list(filter(lambda shiftby: shiftby in dlist2biglist,Shiftby))
    Frequencylist=[dlist2biglist.count(i) for i in Shiftbylist]#this step actually is not necessary for calculating primers
    if not Frequencylist:
    ##do something see below
        Frequencylist=[0]*len(Shiftbylist)
        dGprimlist=[0]*len(Shiftbylist)
        dGSelfdimer_mean=0
    else:
        dlist3=list(map(lambda z: [i for i in range(len(dlist2)) if z in dlist2[i]],Shiftbylist))
        seqlist=list(map(lambda x: [primer1[o] for o in x],dlist3))
        seqlist=[''.join(i) for i in seqlist]
        dGvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['dGvalue'] for i in seqlist]
        dGprimlist=dGvaluelist
        dGSelfdimer_mean=round(np.mean([i for i in dGprimlist if i!=0]),2)
    return (dGSelfdimer_mean)


# In[7]:


'''Drawing and calculating Primer dimer'''
def drawingboxesprimerdimer(box1,box3,box2,drawinglist,primer1,primer2):
    for n in range(len(drawinglist[1])):
        box2=box2[:drawinglist[1][n]]+'|'+box2[(drawinglist[1][n]+1):]
    move=' '*abs(drawinglist[0])
    if drawinglist[0]<0:
        box1=primer1
        box3=move+primer2
    if drawinglist[0]>0:
        box1=move+primer1
        box3=primer2
        box2=move+box2
    if drawinglist[0]==0:
        box1=primer1
        box3=primer2
    return ((box1,box2,box3),drawinglist[0],drawinglist[2],drawinglist[3])
def DrawPrimerdimercalcopt(primer1,primer2,name1,name2,file,logfile,Na,Mg,dNTP,oligo):
    file.write('####Combination: {} and {} ######'.format(name1,name2))
    print('####Combination: {} and {} ######'.format(name1,name2))
    while (len(name1)+len(name2)>30):
        name1=name1[:-1]
        name2=name2[:-1]
    Shiftby=list(range(-len(primer1),len(primer1)+1))
    dlist2=list(map(lambda h: [i-h for i,x in enumerate(primer2) if x==(reverseComplement(primer1[h]))],range(len(primer1))))
    dlist2biglist=[x for sublist in dlist2 for x in sublist]
    Shiftbylist=list(filter(lambda shiftby: shiftby in dlist2biglist,Shiftby))
    Frequencylist=[dlist2biglist.count(i) for i in Shiftbylist]#this step actually is not necessary for calculating primers
    if not Frequencylist:
    ##do something see below
        Frequencylist=[0]*len(Shiftbylist)
        dGprimlist=[0]*len(Shiftbylist)
    else:
        dlist3=list(map(lambda z: [i for i in range(len(dlist2)) if z in dlist2[i]],Shiftbylist))
        seqlist=list(map(lambda x: [primer1[o] for o in x],dlist3))
        seqlist=[''.join(i) for i in seqlist]
        dGvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['dGvalue'] for i in seqlist]
        Tmvaluelist=[SantaLuciacalc(i,Na,Mg,dNTP,oligo)['Tm'] for i in seqlist]
        dGprimlist=dGvaluelist
        #''''''drawing''''''
        box2=[' '*len(primer1) for i in Shiftbylist]
        box1=box3=['' for i in Shiftbylist]
        drawinglist=list(zip(Shiftbylist,dlist3,dGvaluelist,Tmvaluelist))
        draw=list(map(drawingboxesprimerdimer,box1,box3,box2,drawinglist,it.repeat(primer1),it.repeat(primer2)))
        #print boxes
        for i in range(len(draw)):
            print('\n')
            print('\n'.join((draw[i][0])))
            print('It has shifted by {0} position, and this structure has delta G value of {1}, and Tm value of {2}'.format(draw[i][1],draw[i][2],draw[i][3]))
            file.write('\n')
            file.write('\n'.join((draw[i][0])))
            file.write('It has shifted by {0} position, and this structure has delta G value of {1}, and Tm value of {2}'.format(draw[i][1],draw[i][2],draw[i][3]))
        number4=pd.DataFrame(data={'Shiftby':Shiftbylist,'Frequencylist': Frequencylist, 'dGprim': dGprimlist,'Tmprim':Tmvaluelist}
                             ,columns=['Shiftby','Frequencylist','dGprim','Tmprim'])
        number4.to_excel(logfile,sheet_name='{}&{}'.format(name1,name2),index=False,freeze_panes=[1,1])


# In[ ]:



import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
class Window:
    def __init__(self,master):
        Bigframe=tk.Frame(master,width=650,height=650)
        Bigframe.grid()
        Bigframe.grid_propagate(False)
        ##Top Frame
        #"""
        topFrame=tk.Frame(Bigframe,width=650,height=400,padx=20,pady=100)
        topFrame.grid()
        topFrame.grid_propagate(False)
        textboxFrame=tk.Frame(topFrame,width=400,height=200)
        textboxFrame.grid()
        textboxFrame.grid_propagate(False)
        self.textbox=tk.Text(textboxFrame,bd='4',height=12,width=49,relief='raised')
        self.textbox.grid()
        self.outputprimers=tk.Button(topFrame,text='List Primers and Save',command=self.Executecalc)
        self.outputprimers.place(x=150,y=210)
        self.Chooseprimers=tk.Button(Bigframe,text='Choose from a file',command=self.browsetxttop)
        self.Chooseprimers.place(x=25,y=60)
        
        PropertiesLabelR=tk.Label(Bigframe,text='Forward Primer search')
        PropertiesLabelR.place(x=464,y=55)
        PropertiesLabelFromForward=tk.Label(Bigframe,text='From')
        PropertiesLabelFromForward.place(x=470,y=70)#
        PropertiesLabelToForward=tk.Label(Bigframe,text='To')
        PropertiesLabelToForward.place(x=557,y=70)#
        
        
        self.EntryForwardFrom=tk.Entry(Bigframe,width=5)
        self.EntryForwardFrom.place(x=468,y=90)
        self.EntryForwardTo=tk.Entry(Bigframe,width=5)
        self.EntryForwardTo.place(x=548,y=90)
                
        PropertiesLabelR=tk.Label(Bigframe,text='Reverse Primer search')
        PropertiesLabelR.place(x=467,y=113)
        
        PropertiesLabelFromReverse=tk.Label(Bigframe,text='From')
        PropertiesLabelFromReverse.place(x=470,y=130)#
        PropertiesLabelToReverse=tk.Label(Bigframe,text='To')
        PropertiesLabelToReverse.place(x=557,y=130)#
        
        self.EntryReverseFrom=tk.Entry(topFrame,width=5)
        self.EntryReverseFrom.place(x=450,y=55)
        self.EntryReverseTo=tk.Entry(topFrame,width=5)
        self.EntryReverseTo.place(x=530,y=55)
        
        
        self.EntryMinbp=tk.Entry(topFrame,width=5)
        self.EntryMinbp.place(x=450,y=100)
        self.EntryMaxbp=tk.Entry(topFrame,width=5)
        self.EntryMaxbp.place(x=530,y=100)
        PropertiesLabelMinbp=tk.Label(Bigframe,text='Min length')
        PropertiesLabelMinbp.place(x=460,y=175)#
        PropertiesLabelMaxbp=tk.Label(Bigframe,text='Max length')
        PropertiesLabelMaxbp.place(x=540,y=175)#
        self.EntryMinTa=tk.Entry(topFrame,width=5)
        self.EntryMinTa.place(x=450,y=150)
        self.EntryMaxTa=tk.Entry(topFrame,width=5)
        self.EntryMaxTa.place(x=530,y=150)
        PropertiesMinta=tk.Label(Bigframe,text='Min Ta')
        PropertiesMinta.place(x=465,y=230)
        PropertiesMaxta=tk.Label(Bigframe,text='Max Ta')
        PropertiesMaxta.place(x=545,y=230)
        
        self.EntryNa=tk.Entry(Bigframe,width=5)
        self.EntryNa.place(x=468,y=300)
        self.EntryMg=tk.Entry(Bigframe,width=5)
        self.EntryMg.place(x=548,y=300)
        self.EntrydNTP=tk.Entry(Bigframe,width=5)
        self.EntrydNTP.place(x=468,y=342)
        self.Entryoligo=tk.Entry(Bigframe,width=5)
        self.Entryoligo.place(x=548,y=342)
        PropertiesLabelNa=tk.Label(Bigframe,text='[Na+]')
        PropertiesLabelNa.place(x=468,y=277)
        PropertiesLabelMg=tk.Label(Bigframe,text='[Mg2+]')
        PropertiesLabelMg.place(x=549,y=277)
        PropertiesLabeldNTP=tk.Label(Bigframe,text='[dNTP]')
        PropertiesLabeldNTP.place(x=468,y=320)
        PropertiesLabeloligo=tk.Label(Bigframe,text='[oligo]')
        PropertiesLabeloligo.place(x=549,y=320)
        self.EntryNa.insert('end',50)
        self.EntryMg.insert('end',0.0)
        self.EntrydNTP.insert('end',0.0)
        self.Entryoligo.insert('end',50)
        self.EntryMinbp.insert('end',18)
        self.EntryMaxbp.insert('end',24)
        self.EntryMinTa.insert('end',60)
        self.EntryMaxTa.insert('end',70)
        """
        Bottom Frame
        """
        self.bottomtextbox1=tk.Text(Bigframe,bd='4',height=12,width=20,relief='raised')
        self.bottomtextbox1.place(x=25,y=360)        
        self.bottomtextbox2=tk.Text(Bigframe,bd='4',height=12,width=20,relief='raised')
        self.bottomtextbox2.place(x=250,y=360)
        self.Chooseprimerbottom1=tk.Button(Bigframe,text='Choose from a file',command=self.browsetxtbottomF)
        self.Chooseprimerbottom1.place(x=60,y=580)
        self.Chooseprimerbottom2=tk.Button(Bigframe,text='Choose from a file',command=self.browsetxtbottomR)
        self.Chooseprimerbottom2.place(x=285,y=580)
        self.SelfdimerandHairpin=tk.Button(Bigframe,text='Self dimer and Hairpin \n calculation',command=self.Hairpinselfdimercalc)
        self.SelfdimerandHairpin.place(x=470,y=400)
        self.Primerdimer=tk.Button(Bigframe,text='Primer dimer \n calculation',command=self.Primerdimercalc)
        self.Primerdimer.place(x=500,y=500)
        ##STATUS BAR###
        self.status=tk.Label(Bigframe,text='Status: Doing nothing')
        self.status.place(x=400,y=0)
        ##HovertextHelpline###
        self.Helpline=tk.Label(Bigframe,text='Help: ',bg='white',justify='left')
        self.Helpline.place(x=0,y=0)
        self.Help0=tk.Label(Bigframe,text='?')
        self.Help0.place(x=523,y=90)
        self.Help1=tk.Label(Bigframe,text='?')
        self.Help1.place(x=523,y=155)
        self.Help2=tk.Label(Bigframe,text='?')
        self.Help2.place(x=523,y=200)
        self.Help3=tk.Label(Bigframe,text='?')
        self.Help3.place(x=523,y=250)
        self.Help4=tk.Label(Bigframe,text='?')
        self.Help4.place(x=220,y=450)
        self.Help5=tk.Button(Bigframe,text='How to interpret my results?',bg='white',command=self.Help5line)
        self.Help5.place(x=470,y=580)
        self.Help6=tk.Label(Bigframe,text='?')
        self.Help6.place(x=523,y=320)
        self.Help0.bind('<Enter>',self.Help1onEnter)
        self.Help1.bind('<Enter>',self.Help1onEnter)
        self.Help2.bind('<Enter>',self.Help2onEnter)
        self.Help3.bind('<Enter>',self.Help3onEnter)
        self.Help4.bind('<Enter>',self.Help4onEnter)
        self.Help5.bind('<Enter>',self.Help5onEnter)
        self.Help6.bind('<Enter>',self.Help6onEnter)
        self.outputprimers.bind('<Enter>',self.HelpoutputprimersonEnter)
        self.bottomtextbox1.bind('<Enter>',self.Helptextbox)
        self.bottomtextbox2.bind('<Enter>',self.Helptextbox)
        self.textbox.bind('<Enter>',self.Helptextbox)
        self.SelfdimerandHairpin.bind('<Enter>',self.HelpSelfdimerandHairpinonEnter)
        self.Primerdimer.bind('<Enter>',self.HelpPrimerdimeronEnter)
        self.Help0.bind('<Leave>',self.HelplineonLeave)
        self.Help1.bind('<Leave>',self.HelplineonLeave)
        self.Help2.bind('<Leave>',self.HelplineonLeave)
        self.Help3.bind('<Leave>',self.HelplineonLeave)
        self.Help4.bind('<Leave>',self.HelplineonLeave)
        self.Help5.bind('<Leave>',self.HelplineonLeave)
        self.Help6.bind('<Leave>',self.HelplineonLeave)
        self.outputprimers.bind('<Leave>',self.HelplineonLeave)
        self.textbox.bind('<Leave>',self.HelplineonLeave)
        self.bottomtextbox1.bind('<Leave>',self.HelplineonLeave)
        self.bottomtextbox2.bind('<Leave>',self.HelplineonLeave)
        self.SelfdimerandHairpin.bind('<Leave>',self.HelplineonLeave)
        self.Primerdimer.bind('<Leave>',self.HelplineonLeave)
        #Hovertext-end##
    def Help1onEnter(self, event):
        self.Helpline.configure(text='Help: From which to which base pairs in 5-3 direction \n the program needs to search primers?')
    def Help2onEnter(self, event):
        self.Helpline.configure(text='Help: Input Mininum and Maximum length of primers, Default values are : 18-24')
    def Help3onEnter(self, event):
        self.Helpline.configure(text='Help: Input Minimum and Maximum Annealing Temperature (estimated values),\n Default values are MinTa-60C, MaxTa- 70C')
    def Help4onEnter(self, event):
        self.Helpline.configure(text='Help: For Self-dimer and Haipin Structure, user can input primers sequences in both or either of \n'+' boxes on the left and right.'+
                               'For Primer dimer, user must input primers sequences in both boxes, \npreferably Forward primers on one side and Reverse Primers on the other')    
    def Help5onEnter(self, event):
        self.Helpline.configure(text='Help: Read the text on command terminal')
    def Help5line(self):
        print(SantaLuciaNN)
        print('The program utilizes Thermodynamic data, specified by SantaLucia et al (1998) to calculate '+
             'melting Temperature and Secondary Structure stabilities. For specific description and complete algorithm please '+
             'refer to the source code and study paper.\n'+
             'In the process of creating this program, the study has look at many different primer design programs '+
             'available online such as Primer3, Biophp, Oligo Analyzer (IDTDNA), OligoCalculator(Sigma Aldrich) and many others. '+
             'While all of the more popular ones uses nearest neighbour (NN) model, not all of them agree on the same formula, '+
             'parameters, or even salt correction factors, which can have great difference in the final results. '+
              'This problem is eventually further exarcebated when these programs are not open-sourced, or even worse, difficult to understand '+
             'Ultimately, while this program has attempted to find the best possible algorithm, it by no means should be seen as better or worse '+
             'than other programs. In fact it is best to use this program in conjunction with other available and up to date programs to get the most informative decision. '+
             '\n \n \t Hai Le, LSHTM 2018-2019, Michael Miles group')        
    def Help6onEnter(self,event):
        self.Helpline.configure(text='Help: Input [salt] concentration. Default values for [Na+] is 50 milimolar, [Mg] is 0.0 milimolar,\n'+
                               '[dNTP] is 0.0 milimolar, [oligo] is 50 nanomolar')
    def HelplineonLeave(self, event):
        self.Helpline.configure(text="Help: ")
    def HelpoutputprimersonEnter(self,event):
        self.Helpline.configure(text='Help: This option allows the user to get a list of all possible primers within the defined parameters.\n'+
                                'Each primer will receive Tm, GC content, Length, Single Base Stretch, \n'+
                                'GC clamp, Mean dG self primer, and Tm of Hairpin structure closest to defined annealing temperature.\n '+
                               'The output will be saved in an Excel (.xlsx) file. Different genes will be saved under different sheet name')
    def HelpSelfdimerandHairpinonEnter(self,event):
        self.Helpline.configure(text='Help: This option allows user to get the visualization of all possible Self-dimer and Hairpin Structures\n'+
                               'along with their structure stability values, entropy(dG), enthalpy(dH) and melting Temperature (Tm).\n '+
                               'All secondary structure visualization will be saved in Text (.txt) file, \n'+
                               'Summary tables will be saved in Excel (.xlsx) file\n'+
                               'Note: each sheet in Excel must have different names, meaning each input sequence must have different name\n'+
                               'otherwise it will not save correctly')
    def HelpPrimerdimeronEnter(self,event):
        self.Helpline.configure(text='Help: This option allows user to get the visualization of all possible Primer dimer combinations across input primers\n'+
                               'along with their structure stability values ofs entropy(dG)\n'+
                               'All secondary structure visualization will be saved in Text (.txt) file,\n'+
                               'Summary tables will be saved in Excel (.xlsx) file\n'+
                               'Note: each sheet in Excel must have different names, meaning each input sequence must have different name\n'+
                               'otherwise it will not save correctly',justify='left')
    def Helptextbox(self,event):
        self.Helpline.configure(text='Help: User can either enter more than one sequences or choose from a file.'+
                                '\nSequences must be in FASTA form and separated by new lines')
    
    
    def browsetxttop(self):
        filename=browsefile()
        sequence=open(filename,'r').read()
        self.textbox.insert(tk.END,sequence)
    
    def browsetxtbottomF(self):
        filename=browsefile()
        sequence=open(filename,'r').read()
        self.bottomtextbox1.insert(tk.END,sequence)
    
    def browsetxtbottomR(self):
        filename=browsefile()
        sequence=open(filename,'r').read()
        self.bottomtextbox2.insert(tk.END,sequence)
        
    def Executecalc(self):
        #self.outputprimers['state']='disable'
        self.status.configure(text='Status: Running...')
        outputfilepath=file_save(defaultextension='.xlsx')
        seq=self.textbox.get('1.0','end-1c')
        seq=seq.upper()
        ForwardFrom=int(self.EntryForwardFrom.get())
        ForwardTo=int(self.EntryForwardTo.get())
        ReverseFrom=int(self.EntryReverseFrom.get())
        ReverseTo=int(self.EntryReverseTo.get())
        minbp=int(self.EntryMinbp.get())
        maxbp=int(self.EntryMaxbp.get())
        minta=float(self.EntryMinTa.get())
        maxta=float(self.EntryMaxTa.get())
        Na=float(self.EntryNa.get())
        Na=Na*1e-3
        Mg=float(self.EntryMg.get())
        Mg=Mg*1e-3
        dNTP=float(self.EntrydNTP.get())
        dNTP=dNTP*1e-3
        oligo=float(self.Entryoligo.get())
        oligo=oligo*1e-9
        averageta=(minta+maxta)/2
        nameindex=fastatextprocessing(seq)['name']
        sequencelist=fastatextprocessing(seq)['sequencelist']
        logfile=pd.ExcelWriter(outputfilepath,engine='xlsxwriter')
        for i in range(len(nameindex)):
            if nameindex[i] == nameindex[-1]:
                seq=DNAString(sequencelist[nameindex[i]+1:])
            else:
                seq=DNAString(sequencelist[nameindex[i]+1:nameindex[i+1]])
            seqname=sequencelist[nameindex[i]].replace('>','')
            reverseq=reverseComplement(seq)
            e=pd.DataFrame()
            d=pd.DataFrame()
            bp=list(range(minbp,maxbp+1))
            for a in bp:
                root.update()
                pos=ForwardFrom-1
                while pos+a<=ForwardTo-1:
                    primer=seq[pos:pos+a]
                    pos=pos+1
                    #print('It is {} out of {}'.format(primercount,totalprimer))
                    data={'Primers':[primer],
                     'Tm':[SantaLuciacalc(primer,Na,Mg,dNTP,oligo)['Tm']],
                     'GC':[GCcalc(primer)],
                          'Length':[len(primer)],
                          'Pos':[pos],
                     'Single_Base_Stretch':[singlebasestretch(primer)],
                     'GC_clamp':[GCclamp(primer)],
                        'MeandG_Self_dimer':[secondarystructurecalcSelfdimerHairpinopt(primer,averageta,Na,Mg,dNTP,oligo)['dGSelfdimer_mean']],
                          'WorstTmHairpin':[secondarystructurecalcSelfdimerHairpinopt(primer,averageta,Na,Mg,dNTP,oligo)['WorstTmHairpin']],
                        'Direction':['Forward']}
                    print (data)
                    e=e.append(pd.DataFrame(data,columns=['Primers','Tm','GC','Length','Pos','Single_Base_Stretch','GC_clamp','MeandG_Self_dimer','WorstTmHairpin','Direction']))
            for b in bp:
                root.update()
                pos=len(seq)-ReverseTo
                while pos+b<=(len(seq)-ReverseFrom):
                    primer=reverseq[pos:pos+b]
                    #print('It is {} out of {}'.format(primercount,totalprimer))
                    data={'Primers':[primer],
                     'Tm':[SantaLuciacalc(primer,Na,Mg,dNTP,oligo)['Tm']],
                     'GC':[GCcalc(primer)],
                        'Length':[len(primer)],
                          'Pos':[len(seq)-(pos+b)+1],
                     'Single_Base_Stretch':[singlebasestretch(primer)],
                     'GC_clamp':[GCclamp(primer)],
                          'MeandG_Self_dimer':[secondarystructurecalcSelfdimerHairpinopt(primer,averageta,Na,Mg,dNTP,oligo)['dGSelfdimer_mean']],
                          'WorstTmHairpin':[secondarystructurecalcSelfdimerHairpinopt(primer,averageta,Na,Mg,dNTP,oligo)['WorstTmHairpin']],
                         'Direction':['Reverse']}
                    pos=pos+1
                    print (data)
                    d=d.append(pd.DataFrame(data,columns=['Primers','Tm','GC','Length','Pos','Single_Base_Stretch','GC_clamp','MeandG_Self_dimer','WorstTmHairpin','Direction']))
            e.to_excel(logfile,sheet_name='{}_Forward'.format(seqname),index=False,freeze_panes=[1,1])
            d.to_excel(logfile,sheet_name='{}_Reverse'.format(seqname),index=False,freeze_panes=[1,1])
        print('One log file has been saved at {}'.format(outputfilepath))
        self.status.configure(text='Status: Done.')

    def Hairpinselfdimercalc(self):
        self.status.configure(text='Status: Running...')
        outputfilepathtxt=file_save(defaultextension='.txt')
        outputnumber3filepath=outputfilepathtxt.replace('txt','xlsx')
        logfile=pd.ExcelWriter(outputnumber3filepath,engine='xlsxwriter')
        seq1=self.bottomtextbox1.get('1.0','end-1c')
        seq2=self.bottomtextbox2.get('1.0','end-1c')
        seq1=seq1.upper()
        seq2=seq2.upper()
        seq=seq1+'\n'+seq2
        nameindex=fastatextprocessing(seq)['name']
        sequencelist=fastatextprocessing(seq)['sequencelist']
        outputfilepath=open(outputfilepathtxt,'w')
        combinedtable=pd.DataFrame()
        Na=float(self.EntryNa.get())
        Na=Na*1e-3
        Mg=float(self.EntryMg.get())
        Mg=Mg*1e-3
        dNTP=float(self.EntrydNTP.get())
        dNTP=dNTP*1e-3
        oligo=float(self.Entryoligo.get())
        oligo=oligo*1e-9
        minta=float(self.EntryMinTa.get())
        maxta=float(self.EntryMaxTa.get())
        averageta=(minta+maxta)/2
        for i in range(len(nameindex)):
            if nameindex[i] == nameindex[-1]:
                seq=DNAString(sequencelist[nameindex[i]+1:])
            else:
                seq=DNAString(sequencelist[nameindex[i]+1:nameindex[i+1]])
            seqname=sequencelist[nameindex[i]].replace('>','')
            DrawSingleSecStructureSelfdimerHairpinopt(seq,seqname,outputfilepath,logfile,Na,Mg,dNTP,oligo)
            data={'Name':[seqname],
                 'Primer':[seq],
                 'Tm':[SantaLuciacalc(seq,Na,Mg,dNTP,oligo)['Tm']],
                 'GC':[GCcalc(seq)],
                 'Length':[len(seq)],
                 'Single_Base_Stretch':[singlebasestretch(seq)],
                 'GC_clamp':[GCclamp(seq)],
                 'MeandG_Self_dimer':[secondarystructurecalcSelfdimerHairpinopt(seq,averageta,Na,Mg,dNTP,oligo)['dGSelfdimer_mean']],
                 'WorstTmHairpin':[secondarystructurecalcSelfdimerHairpinopt(seq,averageta,Na,Mg,dNTP,oligo)['WorstTmHairpin']]}
            print(data)
            combinedtable=combinedtable.append(pd.DataFrame(data,columns=['Name','Primer','Tm','GC','Length','Single_Base_Stretch',
                                                                         'GC_clamp','MeandG_Self_dimer','WorstTmHairpin']))
        combinedtable.to_excel(logfile,sheet_name='tablesummary',index=False,freeze_panes=[1,0])
        outputfilepath.close()
        print('Two log files have been saved: {} and {}'.format(outputfilepathtxt,outputnumber3filepath))
        self.status.configure(text='Status: Done.')

    def Primerdimercalc(self):
        self.status.configure(text='Status: Running...')
        outputfilepathtxt=file_save(defaultextension='.txt')
        outputnumber4filepath=outputfilepathtxt.replace('txt','xlsx')
        seqF=self.bottomtextbox1.get('1.0','end-1c')
        seqR=self.bottomtextbox2.get('1.0','end-1c')
        seqF=seqF.upper()
        seqR=seqR.upper()
        logfile=pd.ExcelWriter(outputnumber4filepath,engine='xlsxwriter')
        outputfilepath=open(outputfilepathtxt,'w')
        combo=combinationlist(seqF,seqR)
        combinedtable=pd.DataFrame()
        Na=float(self.EntryNa.get())
        Na=Na*1e-3
        Mg=float(self.EntryMg.get())
        Mg=Mg*1e-3
        dNTP=float(self.EntrydNTP.get())
        dNTP=dNTP*1e-3
        oligo=float(self.Entryoligo.get())
        oligo=oligo*1e-9
        minta=float(self.EntryMinTa.get())
        maxta=float(self.EntryMaxTa.get())
        averageta=(minta+maxta)/2
        for i in range(len(combo['namecombo'])):
            name1=combo['namecombo'][i][0]
            name2=combo['namecombo'][i][1]
            primer1=combo['seqcombo'][i][0]
            primer2=combo['seqcombo'][i][1]
            DrawPrimerdimercalcopt(primer1,primer2,name1,name2,outputfilepath,logfile,Na,Mg,dNTP,oligo)
            data={'Name1':[name1],
                  'Primer1':[primer1],
                  'Name2':[name2],
                  'Primer2':[primer2],
                     'Tm1':[SantaLuciacalc(primer1,Na,Mg,dNTP,oligo)['Tm']],
                  'Tm2':[SantaLuciacalc(primer2,Na,Mg,dNTP,oligo)['Tm']],
                     'GC1':[GCcalc(primer1)],
                  'GC2':[GCcalc(primer2)],
                        'Length1':[len(primer1)],
                  'Length2':[len(primer2)],
                     'Single_Base_Stretch1':[singlebasestretch(primer1)],
                  'Single_Base_Stretch2':[singlebasestretch(primer2)],
                     'GC_clamp1':[GCclamp(primer1)],
                  'GC_clamp2':[GCclamp(primer2)],
                          'MeandG_Self_dimer1':[secondarystructurecalcSelfdimerHairpinopt(primer1,averageta,Na,Mg,dNTP,oligo)['dGSelfdimer_mean']],
                  'MeandG_Self_dimer2':[secondarystructurecalcSelfdimerHairpinopt(primer2,averageta,Na,Mg,dNTP,oligo)['dGSelfdimer_mean']],
                          'WorstTmHairpin1':[secondarystructurecalcSelfdimerHairpinopt(primer1,averageta,Na,Mg,dNTP,oligo)['WorstTmHairpin']],
                  'WorstTmHairpin2':[secondarystructurecalcSelfdimerHairpinopt(primer2,averageta,Na,Mg,dNTP,oligo)['WorstTmHairpin']],
                         'MeandG_Primer_dimer':[PDcalc(primer1,primer2,Na,Mg,dNTP,oligo)],
                 'DifferenceTm':[round(abs(SantaLuciacalc(primer1,Na,Mg,dNTP,oligo)['Tm']-SantaLuciacalc(primer2,Na,Mg,dNTP,oligo)['Tm']),2)]}
            print(data)
            combinedtable=combinedtable.append(pd.DataFrame(data,columns=['Name1','Primer1',
                                                  'Name2','Primer2',
                                                  'Tm1','Tm2','GC1','GC2',
                                                  'Length1','Length2',
                                                  'Single_Base_Stretch1','Single_Base_Stretch2',
                                                  'GC_clamp1','GC_clamp2','MeandG_Self_dimer1','MeandG_Self_dimer2',
                                                  'WorstTmHairpin1','WorstTmHairpin2','MeandG_Primer_dimer','DifferenceTm']))
        combinedtable.to_excel(logfile,sheet_name='tablesummary',index=False,freeze_panes=[1,0])
        outputfilepath.close()
        print('Two log files have been saved: {} and {}'.format(outputfilepathtxt,outputnumber4filepath))
        self.status.configure(text='Status: Done.')
    



root=tk.Tk()
root.title('Primer Design program by Le Hai')
seq=Window(root)
#root.geometry('{}x{}'.format(root.winfo_screenwidth(),root.winfo_screenheight()))
#root.geometry('600x600')
root.mainloop()

