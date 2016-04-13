import argparse
import numpy
import preprocess
import warnings

max_iter = 10
stop_criteria = 1e-10


def char2idx(c):
    if c ==' ':
        return 0
    else:
        return ord(c)-ord('A')+1


def log_add(left,right):
    if left == float("-inf") or right == float("-inf"):
        if left == float("-inf") and right == float("-inf"):
            return numpy.log(0)
        elif left == float("-inf"):
            return right
        else:
            return left
    if right <= left:
        return left + numpy.log1p(numpy.exp( right - left))
    else:
        return right + numpy.log1p(numpy.exp(left - right))

def log_compare(left,right):
    if left ==float("-inf") or right == float("-inf"):
        if left == float("-inf") and right == float("-inf"):
            return numpy.log(0)
        elif left == float("-inf"):
            return right
        else:
            return left
    if right <= left:
        return left
    else:
        return right

def forward(num_state,num_time,start_prob,transition_matrix,emit_matrix,line):
    # TODO WRONG INITIALIZATION, SHOULD BE NEGATIVE INFINITY
    # Done!
    alpha_matrix = numpy.log(numpy.zeros((num_state,num_time+1)))

    # TODO initial state
    for j in xrange(num_state):
        alpha_matrix[j,0] = numpy.log(start_prob[j])

    for i in xrange(1,num_time+1):
        c = line[i-1]
        oi = char2idx(c)
        for j in xrange(num_state):
            for k in xrange(num_state):
                right = alpha_matrix[k,i-1] + numpy.log(transition_matrix[k,j]) + numpy.log(emit_matrix[j,oi])
                alpha_matrix[j,i] = log_add(alpha_matrix[j,i],right)
    return alpha_matrix


def viterbi(num_state,num_time,start_prob,transition_matrix,emit_matrix,line):

    viterbi_matrix = numpy.log(numpy.zeros((num_state,num_time+1)))
    backtrack_array = [0 for i in xrange(num_time+1)]

    for j in xrange(num_state):
        viterbi_matrix[j,0] = numpy.log(start_prob[j])

    for i in xrange(1,num_time+1):
        c = line[i-1]
        oi = char2idx(c)
        for j in xrange(num_state):
            for k in xrange(num_state):
                right = viterbi_matrix[k,i-1] + numpy.log(transition_matrix[k,j]) + numpy.log(emit_matrix[j,oi])
                big = log_compare(viterbi_matrix[j,i],right)
                viterbi_matrix[j,i] = big
                if big == right:
                    backtrack_array[i] = k

    return viterbi_matrix,backtrack_array


def eval_alpha(alpha_matrix,t):
    numrow,numcol = alpha_matrix.shape
    res = numpy.log(0)
    for i in xrange(numrow):
        res = log_add(res,alpha_matrix[i,t])
    return res


def eval_beta(beta_matrix,t,start_prob):
    numrow,numcol = beta_matrix.shape
    res = numpy.log(0)
    for i in xrange(numrow):
        right = beta_matrix[i,t] + numpy.log(start_prob[i])
        res = log_add(res,right)
    return res

def backward(num_state,num_time,end_prob,transition_matrix,emit_matrix,line):
    beta_matrix = numpy.log(numpy.zeros((num_state,num_time+1)))

    # TODO initial state
    for j in xrange(num_state):
        beta_matrix[j,num_time] = numpy.log(end_prob[j])

    for i in xrange(num_time-1,-1,-1):
        c = line[i]
        o = char2idx(c)
        for j in xrange(num_state):
            for k in xrange(num_state):
                right = beta_matrix[k,i+1] + numpy.log(transition_matrix[j,k]) + numpy.log(emit_matrix[k,o])
                beta_matrix[j,i] = log_add(beta_matrix[j,i],right)

    return  beta_matrix


def zeta(num_state,num_time,alpha_matrix,beta_matrix,transition_matrix,emit_matrix,line):
    zeta_matrix = numpy.log(numpy.zeros((num_time,num_state,num_state)))
    prob_o = eval_alpha(alpha_matrix,num_time)

    for t in xrange(num_time):
        c = line[t]
        o = char2idx(c)
        for i in xrange(num_state):
            for j in xrange(num_state):
                right = alpha_matrix[i,t] + numpy.log(transition_matrix[i,j]) + numpy.log(emit_matrix[j,o]) + beta_matrix[j,t+1]
                # todo scale to log for zeta
                zeta_matrix[t,i,j] = right - prob_o

    return numpy.exp(zeta_matrix)

def estimate_transition_matrix(zeta_matrix):
    num_time, num_state,num_state2 = zeta_matrix.shape

    denom = zeta_matrix.sum(0)
    nominator = zeta_matrix.sum((0,2),keepdims = True)
    res = (denom/nominator)[0]

    return res

def estimate_emit_matrix(num_state,num_output,zeta_matrix,line):
    num_time,num_state,num_state2 = zeta_matrix.shape

    denom  = [ numpy.zeros((1,num_state)) for i in xrange(num_output)]
    nominator = zeta_matrix.sum((0,1),keepdims = True)[0]

    for t in xrange(num_time):
        c = line[t]
        o = char2idx(c)
        denom[o] += zeta_matrix[t].sum(0,keepdims = True)

    em = numpy.zeros((num_state,num_output))

    for t in xrange(num_output):
        # print "denom: ", denom[t]
        em[:,t] = denom[t]/nominator
        # print (denom[t]/nominator).shape

    return em

def build_start_end_prob(num_state,is_random):
    start_prob = [0 for i in xrange(num_state)]
    end_prob = [0 for i in xrange(num_state)]

    if is_random:
        start_prob = numpy.random.sample((num_state,))
        s = numpy.sum(start_prob)
        start_prob = start_prob/s
        end_prob = numpy.random.sample((num_state,))
        s = numpy.sum(end_prob)
        end_prob = end_prob/s
    else:
        start_prob[1] = 1.0
        end_prob[1] = 1.0

    return start_prob,end_prob

def train_forward_backward_with_em(num_state,num_time,num_output,start_prob,end_prob,is_nature,line):

    # initialize transition and emission matrix
    if is_nature == True:
        tm = transition_matrix(num_state,line)
    # print tm
        em = emission_prob(num_state,num_output,line)
    else:
        tm = transition_matrix_random(num_state,line)
        em = emission_prob_random(num_state,num_output,line)
    # print em
    num_iter = 0

    pc_ll_old = 1
    pc_ll_new = 0
    while True:
        alpha_matrix = forward(num_state,num_time,start_prob,tm,em,line)
        # print alpha_matrix
        beta_matrix = backward(num_state,num_time,end_prob,tm,em,line)
        # print beta_matrix
        zeta_matrix = zeta(num_state,num_time,alpha_matrix,beta_matrix,tm,em,line)
        # print zeta_matrix
        tm = estimate_transition_matrix(zeta_matrix)
        em = estimate_emit_matrix(num_state,num_output,zeta_matrix,line)
        pc_ll_new = eval_alpha(alpha_matrix,num_time)/num_time
        print "pc-ll: ", pc_ll_new
        num_iter += 1
        ratio = abs((pc_ll_new-pc_ll_old)/pc_ll_old)

        if ratio < stop_criteria or num_iter > max_iter:
            pc_ll_beta = eval_beta(beta_matrix,0,start_prob)/num_time
            print "pc-ll-alpha",pc_ll_new
            print "pc-ll-beta",pc_ll_beta
            print "transition: ",tm
            print "emission: ",em
            break
        else:
            pc_ll_old = pc_ll_new

    return tm,em


def log_likelihood_alpha(alpha_matrix):
    numrow,numcol = alpha_matrix.shape
    res = 0
    for i in xrange(numrow):
        res += alpha_matrix[i,numcol-1]

    return numpy.log(res)/(numcol-1)

def log_likelihood_beta(beta_matrix):
    numrow,numcol = beta_matrix.shape
    res = 0
    for i in xrange(numrow):
        res += beta_matrix[i,0]

    return numpy.log(res)/numcol



def transition_matrix(num_state,line):
    vowel_set = preprocess.vowel_set
    non_vowel_set = preprocess.none_vowel_set
    tm = preprocess.transition_prob(vowel_set,non_vowel_set,line)
    tm = numpy.array(tm)
    return tm

def transition_matrix_random(num_state,line):
    sp = numpy.random.sample((num_state,num_state))
    s = numpy.sum(sp,1,keepdims= True)
    sp = sp/s
    return sp

def emission_prob(num_state,num_output,line):
    vowel_set = preprocess.vowel_set
    non_vowel_set = preprocess.none_vowel_set

    hm_vowel = preprocess.emit_prob(vowel_set,line)

    hm_non_vowel = preprocess.emit_prob(non_vowel_set,line)

    em = numpy.zeros((num_state,num_output))


    for t in hm_vowel:
        em[0,char2idx(t)] = hm_vowel[t]

    for t in hm_non_vowel:
        em[1,char2idx(t)] = hm_non_vowel[t]

    return em


def emission_prob_random(num_state,num_output,line):
    em = numpy.zeros((num_state,num_output))
    sp = numpy.random.sample((1,num_output))
    s = numpy.sum(sp)
    sp = sp/s
    em[0,:] = sp

    sp2 = numpy.random.sample((1,num_output))
    s2 = numpy.sum(sp2)
    sp2 = sp2/s2
    em[1,:] = sp2

    return em

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',dest='clean_filename',type=str,required = True)
    args = parser.parse_args()
    line = preprocess.read_clean_file(args.clean_filename)
    num_state = 2
    num_output = 27
    num_time = len(line)
    print num_time
    init_random = False
    is_natute = True


    start_prob,end_prob = build_start_end_prob(num_state,init_random)
    tm,em = train_forward_backward_with_em(num_state,num_time,num_output,start_prob,end_prob,is_natute,line)
    viterbi_matrix, backtrack = viterbi(num_state,num_time,start_prob,tm,em,line)
    print viterbi_matrix
    print backtrack
    print emission_prob_random(num_state,num_output,line)


