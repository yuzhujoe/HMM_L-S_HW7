import argparse
import string
import re

vowel_set = set(['A','E','I','O','U'])
none_vowel_set = set(['B','C','D','F','G','H','J','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z',' '])


def transition_prob(vowelset,nonvowelset,line):
    cnt_v2v = 0
    cnt_v2n = 0
    cnt_n2v = 0
    cnt_n2n = 0
    cnt_v = 0
    cnt_n = 0
    cnt = len(line)-1

    for i in xrange(1,cnt):
        (prev,curr) = line[i-1],line[i]
        if prev in vowel_set and curr in none_vowel_set:
            cnt_v2n += 1.0
            cnt_v += 1.0
        elif prev in vowel_set and curr in vowel_set:
            cnt_v2v += 1.0
            cnt_v += 1.0
        elif prev in none_vowel_set and curr in vowel_set:
            cnt_n2v += 1.0
            cnt_n += 1.0
        elif prev in none_vowel_set and curr in none_vowel_set:
            cnt_n2n += 1.0
            cnt_n += 1.0


    return [(cnt_v2v/cnt_v, cnt_v2n/cnt_v),(cnt_n2v/cnt_n,cnt_n2n/cnt_n)]


def emit_prob(charset,line):
    cnt = 0
    hm = {}
    for c in line:
        if c in charset:
            if c not in hm:
                hm[c] = 1.0
            else:
                hm[c] += 1.0
            cnt += 1

    for c in hm:
        hm[c] /= cnt

    return hm


def clean(filename):

    with open(filename,"r") as f:
        with open(filename[0:-4]+".clean","w") as fout:
            line = f.read()
            line = line.upper()
            line = re.sub("[^A-Z]+"," ",line)
            fout.write(line)

def read_clean_file(filename):
    with open(filename) as f:
        line = f.read()
    return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',dest='input_file',type=str)
    parser.add_argument('-e',dest='emit_type',type=str)
    parser.add_argument('-c',dest='clean_filename',type=str,required = True)
    parser.add_argument('-t',dest='transition_prob',type=str)

    args = parser.parse_args()



    if args.input_file == "True":
        clean(args.input_file)

    line = read_clean_file(args.clean_filename)

    hm = {}

    if args.emit_type!= None:
        if args.emit_type == 'vowel':
            hm = emit_prob(vowel_set,line)
        elif args.emit_type == 'nonvowel':
            hm = emit_prob(none_vowel_set,line)

        print hm

    if args.transition_prob != None:
        print transition_prob(vowel_set,none_vowel_set,line)
