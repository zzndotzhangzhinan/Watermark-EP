import numpy as np
from scipy.stats import gamma, norm, binom
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


def compute_gamma_q(q, check_point):
    qs = []
    for t in check_point:
        qs.append(gamma.ppf(q=q,a=t))
    return np.array(qs)


def compute_ind_q(q, ind_delta, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*(1-ind_delta)+ q*np.sqrt(t*(1-ind_delta)*ind_delta))
    return np.array(qs)


def compute_general_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


def compute_ind_q_new(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


####################################################
##
## Compute test statistics for Gumbel-max watermarks
##
####################################################


def update_array(A):
    #N_trial, Final_T = arr.shape[0]
    B = np.zeros_like(A, dtype=bool)
    for i in range(A.shape[0]):  # 对于每个 trial (行)
        first_true_idx = np.argmax(A[i])  
        if A[i, first_true_idx]:  
            B[i, first_true_idx:] = True  
    return B


def h_ars(Ys,  alpha=0.05):
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_ars_qs = compute_gamma_q(1-alpha, check_points)

    # Compute the test scores
    Ys = np.array(Ys)
    h_ars_Ys = -np.log(1-Ys)
    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)

    results = (cumsum_Ys >= h_ars_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)
    

def h_log(Ys,  alpha=0.05):
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_log_qs = compute_gamma_q(alpha, check_points)

    # Compute the test scores
    Ys = np.array(Ys)
    h_log_Ys = np.log(Ys)
    cumsum_Ys = np.cumsum(h_log_Ys, axis=1)

    results = (cumsum_Ys >= -h_log_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)


def h_ind(Ys, ind_delta=0.5, alpha=0.05):
    # Compute critical values
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_ind_qs = binom.ppf(n=check_points, p = 1-ind_delta, q = 1-alpha)

    # Compute the test scores
    Ys = np.array(Ys)
    h_ind_Ys = (Ys >= ind_delta)
    cumsum_Ys = np.cumsum(h_ind_Ys, axis=1)
    
    results = (cumsum_Ys >= h_ind_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)
    


def h_opt_gum(Ys, delta0=0.2,theo=True, alpha=0.05):
    # Compute critical values
    Ys = np.array(Ys)
    check_points = np.arange(1, 1+Ys.shape[-1])

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

        h_help_qs = compute_general_q(1-alpha, mu, Var, check_points)
    else:
        def find_q(N=500):
            Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
            Simu_Y = f(Null_Ys, delta0)
            Simu_Y = np.cumsum(Simu_Y, axis=1)
            h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)
            return h_help_qs
        
        q_lst = []
        for N in [500] * 10:
            q_lst.append(find_q(N))
        h_help_qs = np.mean(np.array(q_lst),axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    seq_res = update_array(results)
    return np.mean(results,axis=0), np.std(results,axis=0), np.mean(seq_res,axis=0)




def ep_or(Ys, NTP, alpha=0.05):
    Ys = np.array(Ys)
    NTP = np.array(NTP)


    E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)


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






def ep_hp(Ys, hp, alpha=0.05):
    Ys = np.array(Ys)
    hp = np.array(hp)


    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
    E = np.zeros_like(Ys)
    N_trial, T = Ys.shape

    def uniform_Ps(Delta, K=50272):
        nd = np.random.uniform(0.001,Delta)
        generated = np.random.uniform(size=K)
        P = generated/generated[1:].sum() * Delta
        P[0] = 1-Delta
        assert P.max() == 1-Delta and np.abs(np.sum(P)- 1)<= 1e-4 #P.max() == 1-nd and np.abs(np.sum(P)- 1)<= 1e-4
        return P

    for i in range(N_trial):
        for t in range(T):
            if hp[i, t] == 1:
                E[i, t] = 1
            else:
                E[i, t] = np.sum(Ys[i, t] ** (1/ uniform_Ps(hp[i, t])-1))


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





def ep_pte_1(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = (1 - PV + PV*np.log(PV)) / ( PV * (np.log(PV))**2 )

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)

    tilde_E = E#compute_TildE(E)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    
    results = (log_M>=np.log(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)



def ep_pte_1_ada(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = (1 - PV + PV*np.log(PV)) / ( PV * (np.log(PV))**2 )

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
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






def ep_pte_2(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = 2*(1-PV)

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)

    tilde_E = E#compute_TildE(E)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    
    results = (log_M>=np.log(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)


def ep_pte_2_ada(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = 2*(1-PV)

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
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
    M = np.exp(log_M)
    
    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)


def ep_pte_3(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = -np.log(PV)

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)

    tilde_E = E#compute_TildE(E)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    
    results = (log_M>=np.log(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)


def ep_pte_3_ada(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = -np.log(PV)

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
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
    M = np.exp(log_M)
    
    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0), M


def ep_pte_4(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = PV**(-0.5)-1

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)

    tilde_E = E#compute_TildE(E)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    
    results = (log_M>=np.log(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)



def ep_pte_4_ada(Ys, alpha=0.05):
    Ys = np.array(Ys)

    PV = 1 - Ys

    E = PV**(-0.5)-1

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
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



def ep_pte_5(Ys, alpha=0.05):
    Ys = np.array(Ys)


    E = 3*Ys**2

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)

    tilde_E = E#compute_TildE(E)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    
    results = (log_M>=np.log(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)


def ep_pte_5_ada(Ys, alpha=0.05):
    Ys = np.array(Ys)


    E = 3*Ys**2

    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
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
    M = np.exp(log_M)
    
    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)



from Grenander_eg_0 import GrenanderDensity

def GD0(Ys, alpha=0.05):
    Ys = np.array(Ys)
    PV = 1-Ys
    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
    def compute_TildE(PV):
        N_trial, T = PV.shape
        TildE = np.zeros_like(PV)
        for i in range(N_trial):
            for t in range(T):
                if t == 0:
                    l_t = 1
                else:
                    # def obj(x):
                    #     return -np.sum(np.log((1-x) + x * E[i, :t]))
                
                    # res = minimize_scalar(obj, bounds=(0,1), method="bounded")
                    # l_t = res.x
                    g = GrenanderDensity(np.concatenate([[0,1], PV[i, :t]]))
                    l_t = g(PV[i,t])
                    # if l_t == 0:
                    #     l_t = 1
                    #l_t = l_t[-1]
                TildE[i, t] = l_t#(1 - l_t) + l_t * E[i, t]

        return TildE

    tilde_E = compute_TildE(PV)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    M = np.exp(log_M)
    
    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0), M


def GD_bw(Ys, bw=50, alpha=0.05):
    Ys = np.array(Ys)
    PV = 1-Ys
    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
    def compute_TildE(PV):
        N_trial, T = PV.shape
        TildE = np.zeros_like(PV)
        for i in range(N_trial):
            for t in range(T):
                if t == 0:
                    l_t = 1
                else:
                    # def obj(x):
                    #     return -np.sum(np.log((1-x) + x * E[i, :t]))
                
                    # res = minimize_scalar(obj, bounds=(0,1), method="bounded")
                    # l_t = res.x
                    g = GrenanderDensity(np.concatenate([[0,1], PV[i, max(0,t-bw):t]]))
                    l_t = g(PV[i,t])
                    # if l_t == 0:
                    #     l_t = 1
                    #l_t = l_t[-1]
                TildE[i, t] = l_t#(1 - l_t) + l_t * E[i, t]

        return TildE

    tilde_E = compute_TildE(PV)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    M = np.exp(log_M)
    
    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0), M


def two_ep_ave(M1, M2, alpha=0.05):
    M = (M1 + M2)/2

    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)


def three_ep_ave(M1, M2, M3, alpha=0.05):
    M = (M1 + M2 + M3)/3

    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0)



def GD1(Ys, alpha=0.05):
    Ys = np.array(Ys)
    PV = 1-Ys
    #E = np.sum(Ys[:, :, None] ** (1/NTP - 1), axis=-1)
    def compute_TildE(PV):
        N_trial, T = PV.shape
        TildE = np.zeros_like(PV)
        for i in range(N_trial):
            for t in range(T):
                if t == 0:
                    l_t = 1
                else:
                    # l_t = res.x
                    # s = np.concatenate([[0], PV[i, :t]])
                    # s = PV[i, :t]
                    g = GrenanderDensity(np.concatenate([[0,1], PV[i, :t], PV[i, :t]]))
                    # g = GrenanderDensity(np.concatenate([[0,1], np.tile(s, 4)]))
                    l_t = g(PV[i,t])
                    # if l_t == 0:
                    #     l_t = 1
                    #l_t = l_t[-1]
                TildE[i, t] = l_t#(1 - l_t) + l_t * E[i, t]

        return TildE

    tilde_E = compute_TildE(PV)
    log_M =  np.cumsum(np.log(tilde_E), axis=1)
    M = np.exp(log_M)
    
    results = (M>=(1/alpha))
    seq_res = update_array(results)
    return np.mean(seq_res,axis=0), np.std(seq_res,axis=0), np.mean(seq_res,axis=0), M


