#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import json
import random
from scipy.stats import gamma, norm, binom
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

K = 1000
c = 5
Delta = 0.7
Final_T = 700
key = 23333
alpha = 0.05
N_trial = 500#5000
generate_data = False

name = f"ep-simulation/results_data_simu/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-max"
print("Used Delta is:", Delta)


def uniform_Ps(Delta,K=K):
    nd = np.random.uniform(0.001,Delta)
    generated = np.random.uniform(size=K)
    P = generated/generated[1:].sum() * Delta
    P[0] = 1-Delta
    assert P.max() == 1-Delta and np.abs(np.sum(P)- 1)<= 1e-4 #P.max() == 1-nd and np.abs(np.sum(P)- 1)<= 1e-4
    return P
    

def generate_uniform_local(inputs, c, key):
    assert len(inputs) >= c
    #random.seed(tuple(inputs[-(c-1):]+[key]))
    seed_val = hash(tuple(inputs[-(c-1):] + [key]))
    random.seed(seed_val)
    generated = np.random.uniform(size=K)
    return generated


def generate_watermark_text(prompt, T=60, c=5, Delta=0.5, key=1):
    inputs = prompt.copy()
    selected_xis = []
    highest_probs = []
    ntp = []

    for _ in range(T):
        Probs = uniform_Ps(Delta)
        uniform_xi = generate_uniform_local(inputs, c, key)
        next_token = np.argmax(uniform_xi ** (1/Probs))

        inputs.append(next_token)
        selected_xis.append(uniform_xi[next_token])
        highest_probs.append(np.max(Probs))
        ntp.append(Probs)

    return inputs, selected_xis, highest_probs, ntp


## CDF and PDF
def F(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += Probs[k]*x**(1/Probs[k])
    return rho

def f(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += x**(1/Probs[k]-1)
    return rho


## Compute critial values
check_points = np.arange(1,1+Final_T)


def compute_gamma_q(q, check_point):
    qs = []
    for t in check_point:
        qs.append(gamma.ppf(q=q,a=t))
    return np.array(qs)

def compute_ind_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)

h_ars_qs = compute_gamma_q(1-alpha, check_points)
h_log_qs = compute_gamma_q(alpha, check_points)


def update_array(A):
    #N_trial, Final_T = arr.shape[0]
    B = np.zeros_like(A, dtype=bool)
    for i in range(A.shape[0]):  # 对于每个 trial (行)
        first_true_idx = np.argmax(A[i])  
        if A[i, first_true_idx]:  
            B[i, first_true_idx:] = True  
    return B


if generate_data:
    prompts = []
    watermarked_text = []
    highest_probs_lst = []
    Ys = []
    ntp_rec = []

    for trial in tqdm(range(N_trial)):
        prompt = np.random.randint(K, size=c).tolist()
        prompts.append(prompt)
        
        Delta_ins = np.random.uniform(0.001, Delta)

        watermark, generated_Ys, highest_probs, ntp = generate_watermark_text(prompt, T=Final_T, c=c, key=key, Delta=Delta_ins)
        watermarked_text.append(watermark)
        Ys.append(generated_Ys)
        highest_probs_lst.append(highest_probs)
        ntp_rec.append(ntp)

    save_dict = dict()
    save_dict["p"] = np.array(prompts).tolist()
    save_dict["w"] = np.array(watermarked_text).tolist()
    save_dict["y"] = np.array(Ys).tolist()
    save_dict["h"] = np.array(highest_probs_lst).tolist()
    save_dict["n"] = np.array(ntp_rec).tolist()

    json.dump(save_dict, open(name+".json", 'w'))

else:
    save_dict = json.load(open(name+".json", "r"))

    prompts = save_dict["p"] 
    watermarked_text = save_dict["w"] 
    Ys = save_dict["y"]
    highest_probs_lst = save_dict["h"] 
    ntp_rec = save_dict["n"] 


def h_ars(Ys):
    Ys = np.array(Ys)
    h_ars_Ys = -np.log(1-Ys)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_ars_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)


def h_opt(Ys, delta0=0.2,theo=False):
    # This is for the optimal score function
    Ys = np.array(Ys)

    def f(r, delta):
        inte_here = np.floor(1/(1-delta))
        rest = 1-(1-delta)*inte_here
        return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))
    
    h_ars_Ys = f(Ys, delta0)
    
    if theo:
        mu = quad(lambda x: f(x, delta0), 0, 1,epsabs = 1e-10,epsrel=1e-10)
        EX2 = quad(lambda x: f(x, delta0)**2, 0, 1,epsabs = 1e-10,epsrel=1e-10)

        mu, EX2 = mu[0], EX2[0]
        Var = EX2 - mu**2

        h_help_qs = compute_ind_q(1-alpha, mu, Var, check_points)
    else:
        Null_Ys = np.random.uniform(size=(N_trial*2, Final_T))
        Simu_Y = f(Null_Ys, delta0)
        Simu_Y = np.cumsum(Simu_Y, axis=1)
        h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)


def h_log(Ys):
    Ys = np.array(Ys)
    h_log_Ys = np.log(Ys)
    cumsum_Ys = np.cumsum(h_log_Ys, axis=1)
    
    results = (cumsum_Ys >= -h_log_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)


def h_ind(Ys, ind_delta=0.5):
    Ys = np.array(Ys)
    h_ind_Ys = (Ys >= ind_delta)
    cumsum_Ys = np.cumsum(h_ind_Ys, axis=1)
    h_ind_qs = binom.ppf(n=check_points, p = 1-ind_delta, q = 1-alpha)
    results = (cumsum_Ys >= h_ind_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)


def ep_or(Ys, NTP=ntp_rec, lamb_fix=False, lamb=0.99):
    Ys = np.array(Ys)
    NTP = np.array(NTP)

    E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)

    if lamb_fix:
        tilde_E = (1-lamb) + lamb * E
        log_M =  np.cumsum(np.log(tilde_E), axis=1)
    else:
        def compute_TildE(E):
            N_trial, T = E.shape
            TildE = np.zeros_like(E)
            for i in range(N_trial):
                for t in range(T):
                    if t == 0:
                        l_t = 1
                    else:
                        def obj(x):
                            return -np.sum(np.log((1-x) + x * E[i, :t]))
                
                        res = minimize_scalar(obj, bounds=(0,1), method="bounded")
                        l_t = res.x

                    TildE[i, t] = (1 - l_t) + l_t * E[i, t]

            return TildE
        tilde_E = compute_TildE(E)
        log_M =  np.cumsum(np.log(tilde_E), axis=1)
    
    results = (log_M>=np.log(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)




def generate_irr_ntp(T=60, Delta=0.5):
    ntp = []
    for _ in range(T):
        Probs = uniform_Ps(np.random.uniform(0.001, Delta))
        ntp.append(Probs)
    return ntp



from score_functions_simu import ep_pte_2_ada, ep_pte_3_ada, ep_pte_5_ada, GD0, GD_bw, two_ep_ave, three_ep_ave, GD1

##############################################
##
## Check the Type II errors
##
##############################################
result_dict = dict()


result_dict["ars"] = h_ars(Ys)[0].tolist()
result_dict["log"] = h_log(Ys)[0].tolist()

result_dict["opt-0001"] = h_opt(Ys,0.001)[0].tolist()
result_dict["opt-0005"] = h_opt(Ys,0.005)[0].tolist()
result_dict["opt-005"] = h_opt(Ys,0.05)[0].tolist()
result_dict["opt-001"] = h_opt(Ys,0.01)[0].tolist()
result_dict["opt-02"] = h_opt(Ys,0.2)[0].tolist()
result_dict["opt-01"] = h_opt(Ys,0.1)[0].tolist()

result_dict["ind-05"] = h_ind(Ys,0.5)[0].tolist()
result_dict["ind-08"] = h_ind(Ys,0.8)[0].tolist()
result_dict["ind-02"] = h_ind(Ys,0.2)[0].tolist()
result_dict["ind-03"] = h_ind(Ys,0.3)[0].tolist()
result_dict["ind-01"] = h_ind(Ys,0.1)[0].tolist()
result_dict["ind-09"] = h_ind(Ys,0.9)[0].tolist()
result_dict["ind-1/e"] = h_ind(Ys,1/np.exp(1))[0].tolist()

# result_dict["ep_or"] = ep_or(Ys, NTP=ntp_rec, lamb_fix=True, lamb=0.99)[0].tolist()
# result_dict["ep_truentp_ada"] = ep_or(Ys, NTP=ntp_rec, lamb_fix=False, lamb=0.99)[0].tolist()
# result_dict["ep_unknown"] = ep_or(Ys, NTP=generate_irr_ntp(Final_T, Delta), lamb_fix=False, lamb=0.99)[0].tolist()
# result_dict["ep_02a"] = ep_pte_2_ada(Ys, alpha=alpha)[0].tolist()
# ep_fntp_res_02a = ep_pte_2_ada(Ys, alpha=alpha)
# result_dict["ep_02a"] = ep_fntp_res_02a[0].tolist()

ep_fntp_res_03a = ep_pte_3_ada(Ys, alpha=alpha)
result_dict["ep_03a"] = ep_fntp_res_03a[0].tolist()
#result_dict["ep_03a"] = ep_pte_3_ada(Ys, alpha=args.alpha)[0].tolist()


# result_dict["ep_05a"] = ep_pte_5_ada(Ys, alpha=alpha)[0].tolist()
# ep_fntp_res_05a = ep_pte_5_ada(Ys, alpha=alpha)
# result_dict["ep_05a"] = ep_fntp_res_05a[0].tolist()

#result_dict["ep_1"] = ep_or(Ys, NTP=generate_irr_ntp(T=500, Delta=1), alpha=args.alpha)[0].tolist()
# ep_fntp_res_GD0 = GD0(Ys, alpha=alpha)
# result_dict["ep_GD0"] = ep_fntp_res_GD0[0].tolist()

ep_fntp_res_GD1 = GD1(Ys, alpha=alpha)
result_dict["ep_GD1"] = ep_fntp_res_GD1[0].tolist()

result_dict["ep_GD1log"] = two_ep_ave(ep_fntp_res_GD1[3],ep_fntp_res_03a[3],alpha=alpha)[0].tolist()

# ep_fntp_res_GD0_50 = GD_bw(Ys, bw=50, alpha=args.alpha)
# result_dict["ep_GD_bw50"] = ep_fntp_res_GD0_50[0].tolist()

# ep_fntp_res_GD0_100 = GD_bw(Ys, bw=100, alpha=args.alpha)
# result_dict["ep_GD_bw100"] = ep_fntp_res_GD0_100[0].tolist()


# result_dict["ep_GDlog"] = two_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_03a[3],alpha=alpha)[0].tolist()
# result_dict["ep_GDY"] = two_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_02a[3],alpha=alpha)[0].tolist()
# result_dict["ep_GDY**2"] = two_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_05a[3],alpha=alpha)[0].tolist()
# result_dict["ep_GDlogY"] = three_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_02a[3],ep_fntp_res_03a[3],alpha=alpha)[0].tolist()
    #result_dict["ep_1"] = ep_or(Ys, NTP=generate_irr_ntp(T=500, Delta=1), alpha=args.alpha)[0].tolist()

json.dump(result_dict, open(name+"-result"+".json", 'w'))


##############################################
##
## Check the Type I errors
##
##############################################
result_dict = dict()
Null_Ys = np.random.uniform(size=(N_trial, Final_T))
# ars_res = h_ars(Null_Ys)
# result_dict["ars"] = ars_res[0].tolist()


# Null_Ys = np.random.uniform(size=(N_trial, Final_T))

# opt_0_2_res = h_opt(Null_Ys, 0.2)
# result_dict["opt-02"] = opt_0_2_res[0].tolist()

# opt_0_1_res = h_opt(Null_Ys, 0.1)
# result_dict["opt-01"] = opt_0_1_res[0].tolist()

# opt_0_05_res = h_opt(Null_Ys, 0.05)
# result_dict["opt-005"] = opt_0_05_res[0].tolist()

# opt_0_01_res = h_opt(Null_Ys, 0.01)
# result_dict["opt-001"] = opt_0_01_res[0].tolist()

# opt_0_005_res = h_opt(Null_Ys, 0.005)
# result_dict["opt-0005"] = opt_0_005_res[0].tolist()



# Null_Ys = np.random.uniform(size=(N_trial, Final_T))

# log_res = h_log(Null_Ys)
# result_dict["log"] = log_res[0].tolist()

# ind_res = h_ind(Null_Ys, 1/np.exp(1))
# result_dict["ind-1/e"] = ind_res[0].tolist()

# ep_or_res = ep_or(Null_Ys, NTP=ntp_rec, lamb_fix=True, lamb=0.99)
# result_dict["ep_or"] = ep_or_res[0].tolist()

# ep_ada_res = ep_or(Null_Ys, NTP=ntp_rec, lamb_fix=False, lamb=0.99)
# result_dict["ep_truentp_ada"] = ep_ada_res[0].tolist()

# ep_fntp_res = ep_or(Null_Ys, NTP=generate_irr_ntp(Final_T, Delta), lamb_fix=False, lamb=0.99)
# result_dict["ep_unknown"] = ep_fntp_res[0].tolist()

ars_res = h_ars(Null_Ys)
result_dict["ars"] = ars_res[0].tolist()


opt_0_2_res = h_opt(Null_Ys)
result_dict["opt-02"] = opt_0_2_res[0].tolist()

opt_0_1_res = h_opt(Null_Ys)
result_dict["opt-01"] = opt_0_1_res[0].tolist()

opt_0_05_res = h_opt(Null_Ys)
result_dict["opt-005"] = opt_0_05_res[0].tolist()

opt_0_01_res = h_opt(Null_Ys)
result_dict["opt-001"] = opt_0_01_res[0].tolist()

opt_0_005_res = h_opt(Null_Ys)
result_dict["opt-0005"] = opt_0_005_res[0].tolist()

log_res = h_log(Null_Ys)
result_dict["log"] = log_res[0].tolist()

ind_res = h_ind(Null_Ys, 1/np.exp(1))
result_dict["ind-1/e"] = ind_res[0].tolist()


# ep_fntp_res_02a = ep_pte_2_ada(Null_Ys, alpha)
# result_dict["ep_02a"] = ep_fntp_res_02a[0].tolist()


ep_fntp_res_03a = ep_pte_3_ada(Null_Ys, alpha=alpha)
result_dict["ep_03a"] = ep_fntp_res_03a[0].tolist()




# ep_fntp_res_05a = ep_pte_5_ada(Null_Ys, alpha=alpha)
# result_dict["ep_05a"] = ep_fntp_res_05a[0].tolist()

# ep_fntp_res_GD0 = GD0(Null_Ys, alpha=alpha)
# result_dict["ep_GD0"] = ep_fntp_res_GD0[0].tolist()

ep_fntp_res_GD1 = GD1(Null_Ys, alpha=alpha)
result_dict["ep_GD1"] = ep_fntp_res_GD1[0].tolist()

# ep_fntp_res_GD0_50 = GD_bw(Ys, bw=100, alpha=args.alpha)
# result_dict["ep_GD_bw50"] = ep_fntp_res_GD0_50[0].tolist()

# ep_fntp_res_GD0_100 = GD_bw(Ys, bw=200, alpha=args.alpha)
# result_dict["ep_GD_bw100"] = ep_fntp_res_GD0_100[0].tolist()


# ep_fntp_res_GDlog = two_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_03a[3],alpha=alpha)
# result_dict["ep_GDlog"] = ep_fntp_res_GDlog[0].tolist()


# ep_fntp_res_GDY = two_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_02a[3],alpha=alpha)
# result_dict["ep_GDY"] = ep_fntp_res_GDY[0].tolist()

# ep_fntp_res_GDY2 = two_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_05a[3],alpha=alpha)
# result_dict["ep_GDY**2"] = ep_fntp_res_GDY2[0].tolist()

# ep_fntp_res_GDlogY = three_ep_ave(ep_fntp_res_GD0[3],ep_fntp_res_02a[3],ep_fntp_res_03a[3],alpha=alpha)
# result_dict["ep_GDlogY"] = ep_fntp_res_GDlogY[0].tolist()
ep_fntp_res_GD1log = two_ep_ave(ep_fntp_res_GD1[3],ep_fntp_res_03a[3],alpha=alpha)
result_dict["ep_GD1log"] = ep_fntp_res_GD1log[0].tolist()



json.dump(result_dict, open(name+"-null"+".json", 'w'))



##############################################
##
## Check the Sequential Type I errors
##
##############################################

result_dict = dict()
result_dict["ars"] = ars_res[2].tolist()

result_dict["opt-02"] = opt_0_2_res[2].tolist()

result_dict["opt-01"] = opt_0_1_res[2].tolist()

result_dict["opt-005"] = opt_0_05_res[2].tolist()

result_dict["opt-001"] = opt_0_01_res[2].tolist()

result_dict["opt-0005"] = opt_0_005_res[2].tolist()


result_dict["log"] = log_res[2].tolist()

result_dict["ind-1/e"] = ind_res[2].tolist()

# result_dict["ep_or"] = ep_or_res[2].tolist()
# result_dict["ep_truentp_ada"] = ep_ada_res[2].tolist()
# result_dict["ep_unknown"] = ep_fntp_res[2].tolist()

# result_dict["ep_02a"] = ep_fntp_res_02a[2].tolist()


result_dict["ep_03a"] = ep_fntp_res_03a[2].tolist()

# result_dict["ep_05a"] = ep_fntp_res_05a[2].tolist()


# result_dict["ep_GD0"] = ep_fntp_res_GD0[2].tolist()

result_dict["ep_GD1"] = ep_fntp_res_GD1[2].tolist()

# result_dict["ep_GDlog"] = ep_fntp_res_GDlog[2].tolist()
 

# result_dict["ep_GDY"] = ep_fntp_res_GDY[2].tolist()


# result_dict["ep_GDY**2"] = ep_fntp_res_GDY2[2].tolist()


# result_dict["ep_GDlogY"] = ep_fntp_res_GDlogY[2].tolist()
result_dict["ep_GD1log"] = ep_fntp_res_GD1log[2].tolist()

json.dump(result_dict, open(name+"-null-seq"+".json", 'w'))




