#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2 Open “last.txt” file with “read-only” file mode.
infile = open('last.txt', 'r')
infile.close()


# In[2]:


#3 Find the number of characters in the file and assign it to “ans3” variable.
infile = open('last.txt', 'r')
content = infile.read()
ans3 = len(content)
infile.close()


# In[3]:


#4 Find the number of words (including both numbers and names) in the file and assign it to “ans4” variable.
infile = open('last.txt', 'r')
content = infile.read()
words = content.split()
ans4 = len(words)
infile.close()


# In[4]:


#5 Find the number of lines in the file and assign it to “ans5” variable.
infile = open('last.txt', 'r')
lines = infile.readlines()
ans5 = len(lines)
infile.close()


# In[5]:


#6 Count the last names starting with “L” and assign it to “ans6” variable.
infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)
    
ans6 = 0
for i in name:
    if i[0] == 'L':
        ans6 = ans6+1
infile.close()


# In[6]:


#7 Count the last names ending with “E” and assign it to “ans7” variable.
infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)
    
ans7 = 0
for i in name:
    if i[-1] == 'E':
        ans7 = ans7+1

infile.close()


# In[7]:


#8 Count the last names with length 3 and assign it to “ans8” variable.
infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)
    
ans8 = 0
for i in name:
    if len(i) == 3:
        ans8 = ans8+1

infile.close()


# In[8]:


#9 Count the numbers larger than 0.1 and assign it to “ans9” variable.
infile = open('last.txt','r')
num = []
for line in infile:
    each = float(line.split()[1])
    num.append(each)

ans9 = 0
for i in num:
    if i > 0.1:
        ans9 = ans9+1

infile.close()


# In[9]:


#10 Count the numbers smaller than 0.02 and assign it “ans10” variable.
infile = open('last.txt','r')
num = []
for line in infile:
    each = float(line.split()[1])
    num.append(each)

ans10 = 0
for i in num:
    if i < 0.02:
        ans10 = ans10+1

infile.close()


# In[10]:


#11 Get the number next to your last name and assign it to “ans11” variable. 
#    If your last name doesn’t appear, “ans11” should be 0.
infile = open('last.txt','r')
for line in infile:
    lst = line.split()
    if lst[0] == 'KAO':
        ans11 = lst[1]
    else:
        ans11 = 0

infile.close()


# In[11]:


#12 Find the last name that comes last in the dictionary order and assign it to “ans12” variable.
infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)

name.sort()
ans12 = name[-1]

infile.close()


# In[12]:


#13 Find the last name that comes first in the dictionary order and assign it to “ans13” variable.
infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)

name.sort()
ans13 = name[0]

infile.close()


# In[13]:


#14 Find the longest last name and assign it to “ans14” variable.
def length(name):
    return len(name)

infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)

name.sort(key=length)
ans14 = name[-1]

infile.close()


# In[14]:


#15 Find the shortest last name and assign it to “ans15” variable.
def length(name):
    return len(name)

infile = open('last.txt','r')
name = []
for line in infile:
    each = line.split()[0]
    name.append(each)

name.sort(key=length)
ans15 = name[0]

infile.close()


# In[15]:


#16
outfile = open('hw2_answers_41563560_KELLY_KAO.txt','w')

#17
outfile.write('41563560 Kelly Kao kellykao8991@gmail.com\n')

#18
outfile.write('answer 03 = ' + str(ans3)+'\n')
outfile.write('answer 04 = ' + str(ans4)+'\n')
outfile.write('answer 05 = ' + str(ans5)+'\n')
outfile.write('answer 06 = ' + str(ans6)+'\n')
outfile.write('answer 07 = ' + str(ans7)+'\n')
outfile.write('answer 08 = ' + str(ans8)+'\n')
outfile.write('answer 09 = ' + str(ans9)+'\n')
outfile.write('answer 10 = ' + str(ans10)+'\n')
outfile.write('answer 11 = ' + str(ans11)+'\n')
outfile.write('answer 12 = ' + str(ans12)+'\n')
outfile.write('answer 13 = ' + str(ans13)+'\n')
outfile.write('answer 14 = ' + str(ans14)+'\n')
outfile.write('answer 15 = ' + str(ans15)+'\n')

#19
outfile.write('Homework 2 is done!!!')

#20
outfile.close()


# In[ ]:





# In[ ]:




