import math

def __c2n(c):
    return ord(c)-ord('a')

def __n2c(n):
    return chr(ord('a')+n)

def c2n(c):
    lst=[]
    v=0
    lst[:]=c
    length=len(lst)
    for i, c in enumerate(lst):
        v += (__c2n(c))*math.pow(26,(length-i-1))
    return int(v)

def n2c(n):
    v1 = int( n / (26*26) )
    n -= v1 * 26*26
    v2 = int(n / (26))
    n -= v2 * 26 
    v3 = n
    v = [ v1, v2, v3]
    r = [ __n2c(i) for i in v]
    return "".join(r)
    
