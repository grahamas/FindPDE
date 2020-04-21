
linreg = @load LinearRegressor pkg="GLM"
elastic_net = @load ElasticNetRegressor pkg="ScikitLearn"

l0_loss(machine) = sum(fitted_params(machine).coefs .!= 0.0)
l0_loss(vec::AbstractArray) = sum(vec .!= 0.0)

# solve Ax = b
function lstsq(A,b)
    qr(A) \ b
end

function STRidge(X, y, λ, maxit, tol, normalize=2)
    n_obs, n_vars = size(X)
    # TODO: move w_ridge out of this repeat
    w = if λ != 0
        lstsq(X' * X + λ .* I(n_vars), X' * y)
    else
        lstsq(X, y)
    end
    num_relevant = n_vars
    big_idxs = abs.(w) .>= tol
    
    for j in 1:maxit    
        new_big_idxs = abs.(w) .>= tol
        
        # if nothing changed since the last iter, stop
        # FIXME does it mean anything if this is true on the first iter?
        if num_relevant == sum(new_big_idxs)
            break
        else
            num_relevant = sum(new_big_idxs)
        end
        
        # make sure we didn't just lose all the coefficients
        if sum(new_big_idxs) == 0
            if j == 1
                # Tolerance too high: all coefficients set below tolerance
                return w
            else
                break
            end
        end
        big_idxs = new_big_idxs
        
        w[.!big_idxs] .= 0
        if λ != 0
            w[big_idxs] .= lstsq(X[:, big_idxs]' * X[:, big_idxs] + λ .* I(num_relevant), X[:,big_idxs]' * y)
        else
            w[big_idxs] .= lstsq(X[:, big_idxs], y)
        end
    end
    
    if big_idxs != [] # How would big_idxs == [] be possible?
        w[big_idxs] .= lstsq(X[:, big_idxs], y)
    end
    
    return w
end
        

# FIXME: should be cond Φ[:,Not(:u_t)]
# R should have constant column
function train_STRidge(X, Ut, λ, d_tol, maxit=25, STR_iters=10, l0_penalty=0.001*cond(convert(Matrix,X)), p_norm=2, split=0.8, print_best_tol=false)
    # TODO set seed
    
    n_obs, n_vars = size(X)
    train, test = partition(eachindex(Ut), split, shuffle=false)
    X_train, X_test = convert.(Matrix, (X[train,:], X[test, :])) # shouldn't need using MLJ
    Ut_train, Ut_test = convert.(Vector, (Ut[train], Ut[test]))
    
    tol = d_tol
    
    # TODO: test normalizing before lm fit
    normed_X_train = if p_norm != 0
        Mreg = 1.0 ./ norm.([X_train[:,i] for i in 1:n_vars], p_norm)
        X_train .* Mreg'
    else
        X_train
    end
    
    w_best = lstsq(normed_X_train, Ut_train) .* Mreg
    Ut_hat = X_test * w_best
    err_best = norm(Ut_hat .- Ut_test) + l0_penalty*l0_loss(w_best)
    tol_best = 0
            
    # Increase tolerance until test performance decreases
    for iter in 1:maxit

        # Get a set of coefficients and error
        w = STRidge(normed_X_train, Ut_train, λ, STR_iters, tol, p_norm) .* Mreg #FIXME
        err = norm(X_test .* w' .- Ut_test) + l0_penalty*l0_loss(w)
        
        if err <= err_best
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol
        else
            tol = max(0, tol - 2d_tol)
            d_tol = 2d_tol / (maxit / iter)
            tol = tol + d_tol
        end
    end

    if print_best_tol
        print("Optimal tolerance: $tol_best\n")
        print("Optimal error: $err_best\n")
    end

    return w_best
end







    
