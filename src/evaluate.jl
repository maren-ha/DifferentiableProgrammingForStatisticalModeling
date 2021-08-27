"""
functions to evaluate model: 
    evaluate prediction with learned betas, 
    gathering dataframes of prediction errors, 
    calculating statistics
"""

function evaluate_predictions(hospital_list, betas)

    nhospitals = length(hospital_list)

    prederrs_mean = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_modmean = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_zero = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_linreg = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_incrmod = Array{Union{Float64, Missing}}(missing, nhospitals)
    
    for i in 1:nhospitals

        y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences], hospital_list[i][1:end-1,:reported]
        last_currentcases, lastreported = hospital_list[i][end,:currentcases], hospital_list[i][end,:reported]
    
        beta = betas[i]
        
        lastincr = last_currentcases - y[end]
        mean_pred = mean(diff(y))
        modmean_pred = y[end] == 0.0 ? 0.0 : mean_pred
        incrmod_pred = beta[1] + beta[2] * y[end] + beta[3] * z[end]
    
        if isnan(incrmod_pred) || any(abs.(beta) .> 10)
            incrmod_pred = mean_pred
        end
    
        slrdf = DataFrame(dy = diff(y), y = y[2:end], z = z[2:end])
        ols = lm(@formula(dy ~ y + z), slrdf)
        lrbeta = coef(ols)
        #lrbetas[i] = lrbeta
        linreg_pred = lrbeta[1] + lrbeta[2] * y[end] + lrbeta[3] * z[end]
    
        if lastreported 
            prederrs_mean[i] = (lastincr - mean_pred)^2
            prederrs_modmean[i] = (lastincr - modmean_pred)^2
            prederrs_zero[i] = (lastincr - 0)^2
            prederrs_linreg[i] = (lastincr - linreg_pred)^2
            prederrs_incrmod[i] = (lastincr - incrmod_pred)^2
        end
    end
    return prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod
end

function evaluate_predictions(hospital_list, betas, incidences_pred::String="mle")

    nhospitals = length(hospital_list)

    prederrs_mean = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_modmean = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_zero = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_linreg = Array{Union{Float64, Missing}}(missing, nhospitals)
    prederrs_incrmod = Array{Union{Float64, Missing}}(missing, nhospitals)
    
    for i in 1:nhospitals
        if incidences_pred == "mle"
            y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences], hospital_list[i][1:end-1,:reported]
        elseif incidences_pred =="ub"
            y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences_ub], hospital_list[i][1:end-1,:reported]
        elseif incidences_pred == "lb"
            y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences_lb], hospital_list[i][1:end-1,:reported]
        else
            throw("ArgumentError: $(incidences_pred) not supported as value for incidences_pred; it can only take values 'mle', 'ub' or 'lb'")
        end
        last_currentcases, lastreported = hospital_list[i][end,:currentcases], hospital_list[i][end,:reported]
    
        beta = betas[i]
        
        lastincr = last_currentcases - y[end]
        mean_pred = mean(diff(y))
        modmean_pred = y[end] == 0.0 ? 0.0 : mean_pred
        incrmod_pred = beta[1] + beta[2] * y[end] + beta[3] * z[end]
    
        if isnan(incrmod_pred) || any(abs.(beta) .> 10)
            incrmod_pred = mean_pred
        end
    
        slrdf = DataFrame(dy = diff(y), y = y[2:end], z = z[2:end])
        ols = lm(@formula(dy ~ y + z), slrdf)
        lrbeta = coef(ols)
        #lrbetas[i] = lrbeta
        linreg_pred = lrbeta[1] + lrbeta[2] * y[end] + lrbeta[3] * z[end]
    
        if lastreported 
            prederrs_mean[i] = (lastincr - mean_pred)^2
            prederrs_modmean[i] = (lastincr - modmean_pred)^2
            prederrs_zero[i] = (lastincr - 0)^2
            prederrs_linreg[i] = (lastincr - linreg_pred)^2
            prederrs_incrmod[i] = (lastincr - incrmod_pred)^2
        end
    end
    return prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod
end

function getpredictedtrajectory(y, z, r, beta)
    pred_traj = []
    firstseen = false # set to true after skipping potential missings 
    last_y = 0.0 # prevalent cases from previous time point

    for t = 1:length(y)
        if !firstseen 
            if r[t] == 1
                last_y = y[t]
                firstseen = true
            else
                continue
            end
        else # make a prediction for the current increment
            pred_dy = beta[1] + beta[2] * last_y + beta[3] * z[t-1] 
            if r[t] == 1
                dy = y[t] - last_y
                last_y = y[t]
            else
                last_y += pred_dy
            end 
            push!(pred_traj, last_y)
        end
    end
    pred_traj
end

function create_prederrs_df(prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod)
    nonmissinginds = findall(x -> !ismissing(x), prederrs_mean)
    prederrs_df = DataFrame(hospitals = nonmissinginds,
                    meanmodel = Float64.(prederrs_mean[nonmissinginds]),
                    modmeanmodel = Float64.(prederrs_modmean[nonmissinginds]),
                    zeromodel = Float64.(prederrs_zero[nonmissinginds]),
                    linregmodel = Float64.(prederrs_linreg[nonmissinginds]),
                    incrmodel = Float64.(prederrs_incrmod[nonmissinginds])
    )
    return prederrs_df
end

function create_prederrs_df(prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod, hospitalinds)
    nonmissinginds = findall(x -> !ismissing(x), prederrs_mean)
    prederrs_df = DataFrame(hospitals = hospitalinds[nonmissinginds],
                    meanmodel = Float64.(prederrs_mean[nonmissinginds]),
                    modmeanmodel = Float64.(prederrs_modmean[nonmissinginds]),
                    zeromodel = Float64.(prederrs_zero[nonmissinginds]),
                    linregmodel = Float64.(prederrs_linreg[nonmissinginds]),
                    incrmodel = Float64.(prederrs_incrmod[nonmissinginds])
    )
    return prederrs_df
end

function get_summary_df(prederrs_df; save_csv::Bool=false, path_to_results::String="~/", filename::String="")
    models = names(prederrs_df[!,2:end])
    summed_squared_errors = sum.(eachcol(prederrs_df[!,2:end]))
    mses = round.(sum.(eachcol(prederrs_df[!,2:end]))./(nrow(prederrs_df)), digits=3)
    medians = round.(median.(eachcol(prederrs_df[!,2:end])), digits=3)
    firstquartile = round.(quantile.(eachcol(prederrs_df[!,2:end]), 0.25), digits=3)
    thirdquartile = round.(quantile.(eachcol(prederrs_df[!,2:end]), 0.75), digits=3)
    maxvals = round.(maximum.(eachcol(prederrs_df[!,2:end])), digits=3)
    sqerr_df = DataFrame(Model = models, 
                        SSE = summed_squared_errors, MSE = mses, 
                        FirstQuartile = firstquartile,
                        Median = medians, 
                        ThirdQuartile = thirdquartile,
                        Maximum = maxvals
    )
    if save_csv
        CSV.write(string(path_to_results, filename, ".csv"), sqerr_df)
    end
    return sqerr_df
end
