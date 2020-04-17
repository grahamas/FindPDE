

function train_STRidge(R, u_t, λ, d_tol, maxit=25, STR_iters=10, l0_penalty=0.001*cond(R), normalize=2, split=0.8, print_best_tol=false)
    # TODO set seed
    n_obs, n_vars = size(R)
    obsdim = 1
    (train_R, train_ut), (test_R, test_ut) = splitobs(shuffleobs((R,u_t)), split, obsdim)
    
    tol = d_tol
    
    W = 
    