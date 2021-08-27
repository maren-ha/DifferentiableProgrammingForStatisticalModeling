"""
Compare increment model with all individual parameters and baseline models 
"""

#
# Setup  
#
# cd("path/to/Github/repo")
path_to_repo = pwd()
include("src/setup.jl")

#
# Initialize arrays of prediction errors for all hospitals, one for each baseline model
#
nhospitals = length(hospital_list)

prederrs_mean = Array{Union{Float64, Missing}}(missing, nhospitals)
prederrs_modmean = Array{Union{Float64, Missing}}(missing, nhospitals)
prederrs_zero = Array{Union{Float64, Missing}}(missing, nhospitals)
prederrs_linreg = Array{Union{Float64, Missing}}(missing, nhospitals)
prederrs_incrmod = Array{Union{Float64, Missing}}(missing, nhospitals)

#
# Initialize array for betas for increment and linear regression model 
#

betas = collect([0.0; 0.0; 0.0] for i in 1:nhospitals)
lrbetas = collect([0.0; 0.0; 0.0] for i in 1:nhospitals)
λ = 0

#
# track hospitals where estimation does not converge
#

notconvergedcounter = 0
notconvergedinds = []

#
# Fit increment model and all baseline models, separately for each hospital 
#

@showprogress for i = 1:nhospitals
    y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences], hospital_list[i][1:end-1,:reported]
    last_currentcases, lastreported = hospital_list[i][end,:currentcases], hospital_list[i][end,:reported]

    beta = betas[i]
    for epoch=1:150
        curgrad = gradient(arg -> loss(y, z, r, arg) + λ*sum(arg .^2), beta)
        beta .-= [0.001, 0.01, 0.001] .* curgrad[1]
    end
    betas[i] = beta

    lastincr = last_currentcases - y[end]
    mean_pred = mean(diff(y))
    modmean_pred = y[end] == 0.0 ? 0.0 : mean_pred
    incrmod_pred = beta[1] + beta[2] * y[end] + beta[3] * z[end]

    if isnan(incrmod_pred) || any(abs.(beta) .> 10)
        incrmod_pred = mean_pred
        notconvergedcounter += 1
        push!(notconvergedinds, i)
    end

    slrdf = DataFrame(dy = diff(y), y = y[2:end], z = z[2:end])
    ols = lm(@formula(dy ~ y + z), slrdf)
    lrbeta = coef(ols)
    lrbetas[i] = lrbeta
    linreg_pred = lrbeta[1] + lrbeta[2] * y[end] + lrbeta[3] * z[end]

    if lastreported 
        prederrs_mean[i] = (lastincr - mean_pred)^2
        prederrs_modmean[i] = (lastincr - modmean_pred)^2
        prederrs_zero[i] = (lastincr - 0)^2
        prederrs_linreg[i] = (lastincr - linreg_pred)^2
        prederrs_incrmod[i] = (lastincr - incrmod_pred)^2
    end
end

#
# Evaluate: boxplot and summary statistics, reproducing the individual results + baseline models in Table 1
#

prederrs_df = create_prederrs_df(prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod)
rename!(prederrs_df, ["linregmodel" => "Linear regression", "meanmodel" => "Mean model", "incrmodel" => "Increment Model",
                        "modmeanmodel" => "Modified mean model", "zeromodel" => "Zero model"])
#
stacked_df = stack(prederrs_df)
boxplot_localmodels = stacked_df |> @vlplot(
    width=600, height=400,
    mark={:boxplot, extend=2},
    x = {"variable:o", title="Model"},
    y = {"value", scale={type="sqrt", domain=[0,10]}, axis = 0:0.5:10, title="Prediction error (sqrt scale)"}, # one outlier for linear regression removed
    color = {:variable, title="Model"},
    size={value=40}
)
save(string(path_to_results, "boxplot_localmodels.svg"), boxplot_localmodels)

sqerr_df = get_summary_df(prederrs_df, save_csv=true, path_to_results = path_to_results, filename="prederr_stats_localmodels")

#
# Look at hospitals where estimation did not converge
#

notconvergedcounter # = 85

# compare overall numbers of cases in hospitals where estimation did not converge vs the rest 

# hospitals where estimation did not converge
notreported = 0.0
num_cases = 0.0
for i in notconvergedinds 
    y, z, r = hospital_list[i][1:end,:currentcases], hospital_list[i][1:end,:incidences], hospital_list[i][1:end,:reported]
    notreported += sum(.!r)
    num_cases += sum(y)
end
notreported/length(notconvergedinds) # average number of missing reports: 4.0
num_cases/length(notconvergedinds) # average number of cases: 384.4235294117647

# the other hospitals 
notreported = 0.0
num_cases = 0.0
for i in 1:nhospitals
    if i ∈ notconvergedinds
        continue
    end
    y, z, r = hospital_list[i][1:end,:currentcases], hospital_list[i][1:end,:incidences], hospital_list[i][1:end,:reported]
    notreported += sum(.!r)
    num_cases += sum(y)
end
notreported/(nhospitals-notconvergedcounter) # average number of missing reports is comparable: 4.523411371237458
num_cases/(nhospitals-notconvergedcounter) # average number of cases is much smaller: 53.42892976588629

#
# For an examplary hospital where the increment model optimized with differentiable programming
# yields a lower prediction error than a linear regression optimized with standard maximum likelihood,
# compare the loss curves (reproducing Supplementary FIgure 1)
#

i = 124 # pick exemplary hospital 

# 1) get learned parameters and data
beta = betas[i]
lrbeta = lrbetas[i]
y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences], hospital_list[i][1:end-1,:reported]
last_currentcases, lastreported = hospital_list[i][end,:currentcases], hospital_list[i][end,:reported]
lastincr = last_currentcases - y[end]

# 2) evaluate model predictions 
mean_pred = mean(diff(y))
modmean_pred = y[end] == 0.0 ? 0.0 : mean_pred
incrmod_pred = beta[1] + beta[2] * y[end] + beta[3] * z[end]
linreg_pred = lrbeta[1] + lrbeta[2] * y[end] + lrbeta[3] * z[end]
loss_lr = loss_locflinreg(diff(y), y[2:end], z[2:end], lrbeta)
loss_incr = loss(y, z, r, beta)

# plot loss curves
beta1range = -0.3:0.01:0.1
beta2range = -0.4:0.01:0.05
beta3range = -0.2:0.01:0.3

lossdf1 = DataFrame(xs = repeat(collect(beta1range),2),
                ys = cat(collect(loss(y, z, r, [beta1, beta[2], beta[3]]) for beta1 in beta1range),
                        collect(loss_locflinreg(diff(y), y[2:end], z[2:end], [beta1, lrbeta[2], lrbeta[3]]) for beta1 in beta1range)..., dims=1),
                model = repeat(["Increment model", "Linear regression"], inner=length(beta1range))
)
beta1plot = lossdf1 |> @vlplot(:line, 
                x = {"xs", title="Intercept parameter"}, 
                y = {"ys", title="Loss", scale={domain=[0.2,1.0]}}, 
                color = {"model", title="Model", legend={orient="top-left",offset=7}}
) |> save(string(path_to_results, "lossplots_beta1_hospital_$i.svg"))

lossdf2 = DataFrame(xs = repeat(collect(beta2range),2),
                ys = cat(collect(loss(y, z, r, [beta[1], beta2, beta[3]]) for beta2 in beta2range),
                collect(loss_locflinreg(diff(y), y[2:end], z[2:end], [lrbeta[1], beta2, lrbeta[3]]) for beta2 in beta2range)..., dims=1),
                model = repeat(["Increment model", "Linear regression"], inner=length(beta2range))
)
beta2plot = lossdf2 |> @vlplot(:line, 
                x = {"xs", title="Parameter for prevalent cases"}, 
                y = {"ys", title="Loss", scale={domain=[0.2,1.0]}}, 
                color = {"model", title="", legend=false}
) |> save(string(path_to_results, "lossplots_beta2_hospital_$i.svg"))

lossdf3 = DataFrame(xs = repeat(collect(beta3range),2),
                ys = cat(collect(loss(y, z, r, [beta[1], beta[2], beta3]) for beta3 in beta3range),
                collect(loss_locflinreg(diff(y), y[2:end], z[2:end], [lrbeta[1], lrbeta[2], beta3]) for beta3 in beta3range)..., dims=1),
                model = repeat(["Increment model", "Linear regression"], inner=length(beta3range))
)
beta3plot = lossdf3 |> @vlplot(:line, 
                x = {"xs", title="Parameter for predicted incidences"}, 
                y = {"ys", title="Loss", scale={domain=[0.2,1.0]}}, 
                color = {"model", title="", legend=false}
) |> save(string(path_to_results, "lossplots_beta3_hospital_$i.svg"))

#
# additional analysis: compare the predicted trajectories 
#

pred_traj = getpredictedtrajectory(y, z, r, beta)
pred_traj_lr = getpredictedtrajectory(y, z, r, lrbeta)

ntimepoints = nrow(hospital_list[i])
startind = ntimepoints - length(pred_traj)
begin
    df = DataFrame()
    df[!,:date] = hospital_list[i][startind:end-1,:date]
    df[!,:predicted_cases_incr_model] = pred_traj
    df[!,:predicted_cases_locf_linreg] = pred_traj_lr
    df[!,:notreported] = hospital_list[i][startind:end-1,:notreported]
    df
end

p = df |> @vlplot(width = 1000, height = 400, 
                transform = [{calculate = "datum.notreported == true ? 'predicted' : 'reported'",
                            as = "casesreported"}]) +
    @vlplot(mark = {:line, color = "#1f77b4", opacity = 0.7},
        x = {"date:t", title = "Date"},
        y = {"predicted_cases_incr_model:q", title = "Number of prevalent COVID-19 cases"}
    ) +
    @vlplot(data = df[df[!,:notreported].==1,[:predicted_cases_locf_linreg, :date]],
        mark = {:point, filled=true, size=250, opacity = 0.5, color = "#ff7f0e"},
        x = {:date, title = "Date"},
        y = {"predicted_cases_locf_linreg:q", title = "Number of prevalent COVID-19 cases"}
    ) +
    @vlplot(
        mark = {:point, filled=true, size=250, opacity=0.5},
        x = {:date, title = "Date"},
        y = {"predicted_cases_incr_model:q", title = "Number of prevalent COVID-19 cases"},
        color = {"casesreported:n",
                scale = {range = ["#2ca02c", "#1f77b4"]},
                legend = {title = "Prevalent cases reported?"}
                }
    ) + 
    @vlplot(mark = {:line, color = "#1f77b4", opacity = 0.7},
        x = {"date:t", title = "Date"},
        y = {"predicted_cases_locf_linreg:q", title = "Number of prevalent COVID-19 cases"}
)
p |> save(string(path_to_results, "predicted_cases_incr_vs_locf+linreg_$i.svg"))