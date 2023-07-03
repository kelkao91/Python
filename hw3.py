#2
def make_length_wordcount(filename):
    infile = open(filename)
    content = infile.read()
    length_counter = {}
    for word in content.split():
        if len(word) in length_counter:
            length_counter[len(word)] = length_counter[len(word)]+1
        else:
            length_counter[len(word)] = 1
    return length_counter

#3 
def make_word_count(file):
    filename = file+'.txt'
    infile = open(filename)
    content = infile.read()
    lower = content.lower()
    words = lower.split()
    counter1={}
    for word in words:
        if word in counter1:
            counter1[word] += 1
        else:
            counter1[word] = 1
    return counter1

#3
def make_word_count(filename):
    infile = open(filename)
    content = infile.read()
    word_counter = {}
    for word in content.split():
        if word in word_counter:
            word_counter[(word)] = word_counter[(word)]+1
        else :
            word_counter[(word)] = 1
    return word_counter


#4
def analyze_text(filename):
    word_count = make_word_count(filename)
    word_length_count = make_length_wordcount(filename)
    studentid = input('Enter your Student ID:')
    first = input('Enter your first name:')
    last = input('Enter your last name:')
    with open(str(filename)+'_'+'analyzed_'+str(studentid)+'_'+str(first)+'_'+str(last)+'.txt','a') as outfile:
        for x in sorted(word_length_count):
            outfile.write("Words of length " + str(x) + " : " + str(word_length_count[x])+'\n')
        for x in sorted(word_count):
            outfile.write(x + " : " + str(word_count[x])+'\n')
    infile.close()


#5a
analyze_text('nasdaq.txt')

#5b
analyze_text('raven.txt')

#5c
analyze_text('frankenstein.txt')





