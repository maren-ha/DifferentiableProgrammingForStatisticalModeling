"""
Leave some time points Out Validation, see how well model recovers them 
"""

function fit_incr_model_lov(y,z,r,predtraj_mean)
    beta = [0.0; 0.0; 0.0]
    for epoch=1:150
        curgrad = gradient(arg -> loss(y, z, r, arg), beta)
        if !any(isnothing.(curgrad[1]))
            beta .-= [0.001, 0.01, 0.001] .* curgrad[1]
        end
    end
    # predict trajectory
    predtraj_incr = getpredictedtrajectory(y, z, r, beta)
    if any(isnan.(predtraj_incr)) || any(abs.(beta) .> 10)
        predtraj_incr = deepcopy(predtraj_mean)
    end
    return predtraj_incr, beta 
end

function fit_lr_model_lov(y, z, r, predtraj_mean)
    slrdf = DataFrame(dy = diff(y), y = y[2:end], z = z[2:end]);
    ols = lm(@formula(dy ~ y + z), slrdf)
    lrbeta = coef(ols)
    # predict trajectory
    predtraj_lr = getpredictedtrajectory(y, z, r, lrbeta)
    if any(isnan.(predtraj_lr)) || any(abs.(lrbeta) .> 10)
        predtraj_lr = deepcopy(predtraj_mean)
    end
    return predtraj_lr, lrbeta 
end

function modify_r!(r, frac_missing)
    ntimepoints = length(r)
    removeinds = shuffle(2:ntimepoints)[1:Int(floor(frac_missing*ntimepoints))]
    r[removeinds] .= false
    return r
end

function modify_r(true_r, initial_missing_prob, following_missing_prob)
    ntimepoints = length(true_r)
    r = deepcopy(true_r)
    for t in 2:ntimepoints
        if r[t-1]
            t_missing = randn() > initial_missing_prob 
        else 
            t_missing = randn() > following_missing_prob
        end
        #r[t] = !(r[t-1] ? randn() > initial_missing_prob : randn() > following_missing_prob)
        r[t] = !t_missing 
    end
    return r
end

function modify_r(true_r, frac_missing)
    ntimepoints = length(true_r)
    r = deepcopy(true_r)
    removeinds = shuffle(2:ntimepoints)[1:Int(floor(frac_missing*ntimepoints))]
    r[removeinds] .= false
    return r
end

function modify_y(true_y, r)
    y = deepcopy(true_y)
    for t in 2:length(y)
        if r[t] == false
            y[t] = y[t-1]
        end
    end
    return y 
end

function modify_r_wrapper(frac_missing)
    return function (true_r) modify_r(true_r, frac_missing) end
end

function modify_r_wrapper(initial_missing_prob, following_missing_prob)
    return function (true_r) modify_r(true_r, initial_missing_prob, following_missing_prob) end
end

function fitmodels_lov(hospital_list, i, nreps, mod_r)
    true_y, z, true_r = hospital_list[i][:,:currentcases], hospital_list[i][:,:incidences], hospital_list[i][:,:reported]
    ntimepoints = length(true_y)
    prederr_incr = 0.0
    prederr_lr = 0.0
    prederr_zero = 0.0
    prederr_mean = 0.0
    prederr_modmean = 0.0

    betas_reps = [[0.0; 0.0; 0.0] for rep in 1:nreps] 
    lrbetas_reps = [[0.0; 0.0; 0.0] for rep in 1:nreps] 

    for rep in 1:nreps 
        # random flip values in r
        r = mod_r(true_r)

        # replace values in y with r=false with LOCF
        y = modify_y(true_y, r)

        # zero model 
        prederr_zero += sum((true_y[2:end] .- y[2:end]).^2)./(ntimepoints-1)

        # mean model 
        pred_incr = mean(diff(y))
        predtraj_mean = map(t -> r[t] ? y[t] : y[t-1]+pred_incr, 2:length(r))
        prederr_mean += sum((true_y[2:end] .- predtraj_mean).^2)./(ntimepoints-1)

        # modified mean model 
        predtraj_modmean = map(t -> r[t] ? y[t] : (y[t-1] == 0.0 ? 0.0 : y[t-1]+ pred_incr), 2:length(y))
        prederr_modmean += sum((true_y[2:end] .- predtraj_modmean).^2)./(ntimepoints-1)

        # increment model
        predtraj_incr, beta = fit_incr_model_lov(y, z, r, predtraj_mean)
        prederr_incr += sum((true_y[2:end] .- predtraj_incr).^2)./(ntimepoints-1)
        betas_reps[rep] = beta 

        # linear regression with LOCF
        predtraj_lr, lrbeta = fit_lr_model_lov(y, z, r, predtraj_mean)
        prederr_lr += sum((true_y[2:end] .- predtraj_lr).^2)./(ntimepoints-1)
        lrbetas_reps[rep] = lrbeta 

    end
    return prederr_mean, prederr_modmean, prederr_zero, prederr_lr, prederr_incr, betas_reps, lrbetas_reps
end
