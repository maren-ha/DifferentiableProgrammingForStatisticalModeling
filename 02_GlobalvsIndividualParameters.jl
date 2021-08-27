"""
Combinations of global and local parameter estimation and comparison with baseline models 
"""

#
# Setup  
#

# cd("path/to/Github/repo")
path_to_repo = pwd()
include("src/setup.jl")

#
# Try out all combinations of global/individual beta 1,2,3 
#

# first define combinations 
combinations = [["global", "global", "global"],
                ["global", "global", "individual"],
                ["global", "individual", "global"],
                ["global", "individual", "individual"],
                ["individual", "global", "global"],
                ["individual", "global", "individual"],
                ["individual", "individual", "global"],
                ["individual", "individual", "individual"]
]

# function to estimate the parameters for a given combination of global-individual estimation 
function runlocalglobalmix(hospital_list, combination::Array{String,1})

    nhospitals = length(hospital_list)
    betas = [[0.0; 0.0; 0.0] for i in 1:length(hospital_list)]
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]

    @showprogress for epoch in 1:150
        for i = 1:nhospitals
            y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences], hospital_list[i][1:end-1,:reported]
            beta = betas[i]
            curgrad = gradient(arg -> loss(y, z, r, arg), beta)
            beta .-= [0.001, 0.01, 0.001] .* curgrad[1]
            betas[i] = beta
        end

        feasibleinds = findall(x -> (!any(isnan.(x)) && !any(abs.(x) .> 10)), betas) # exclude betas from non-converged fits

        if beta1 == "global"
            for i in feasibleinds#1:length(betas) 
                betas[i][1] = mean(collect(x[1] for x in betas[feasibleinds]))
            end
        end

        if beta2 == "global"
            for i in feasibleinds#1:length(betas) 
                betas[i][2] = mean(collect(x[2] for x in betas[feasibleinds]))
            end
        end

        if beta3 == "global"
            for i in feasibleinds#1:length(betas) 
                betas[i][3] = mean(collect(x[3] for x in betas[feasibleinds]))
            end
        end
    end
    return betas
end

# save the estimated coefficients in a dictionary
betas_all_combs = Dict()

# loop over the combinations and fit all models for each 
for combination in combinations
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]
    @info "starting with new combination: $(beta1), $(beta2), $(beta3)..." 
    betas = runlocalglobalmix(hospital_list, combination)
    betas_all_combs["combination_$(beta1)_$(beta2)_$(beta3)"] = betas
    @info "betas calculated!"
end
# save dictionary of estimated parameters
save(string(path_to_results, "betas_allcombinations.jld2"), Dict("betas_allcombinations" => betas_all_combs))

# if you have done the analyses before, you can start directly from here and load the estimated parameters: 
betasdict = load(string(path_to_results, "betas_allcombinations.jld2"))
betas_all_combs = betasdict["betas_allcombinations"]

#
# Evaluate, reproducing Table 1 
#

# for each combination and each model, get prediction errors and make boxplot
prederrs_df = DataFrame()
for cind in 1:length(combinations)
    combination = combinations[cind] 
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]
    betas = betas_all_combs["combination_$(beta1)_$(beta2)_$(beta3)"]
    prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod = evaluate_predictions(hospital_list, betas)
    nonmissinginds = findall(x -> !ismissing(x), prederrs_mean)
    if cind == 1
        nonmissinginds = findall(x -> !ismissing(x), prederrs_mean)
        prederrs_df = DataFrame(hospital = nonmissinginds,
                        meanmodel = Float64.(prederrs_mean[nonmissinginds]),
                        modmeanmodel = Float64.(prederrs_modmean[nonmissinginds]),
                        zeromodel = Float64.(prederrs_zero[nonmissinginds]),
                        linregmodel = Float64.(prederrs_linreg[nonmissinginds])
        )
    end
    prederrs_df[!,"incrmodel_comb$(cind)"] = Float64.(prederrs_incrmod[nonmissinginds])
end

rename!(prederrs_df, ["linregmodel" => "Linear regression", "meanmodel" => "Mean model", 
                        "modmeanmodel" => "Modified mean model", "zeromodel" => "Zero model"])
for cind in 1:length(combinations)
    combination = combinations[cind] 
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]
    rename!(prederrs_df, "incrmodel_comb$(cind)" => "Increment model: $(beta1[1]), $(beta2[1]), $(beta3[1])")
end

stacked_df = stack(prederrs_df)
boxplots_allcombs = stacked_df |> @vlplot(
    width=600, height=400,
    mark={:boxplot, extend=2},
    x = {"variable:o", title="Model"},
    y = {"value", scale={type="sqrt", domain=[0,20]}, axis={values=0:1:20}, title="Prediction error (sqrt scale)"}, # 1 outlier removed for linear regression
    color = {:variable, title="Model"},
    size={value=40}
)
save(string(path_to_results, "boxplots_allcombinations_mle.svg"), boxplots_allcombs)
# create dataframe with summary statistic (Table 1)

summary_df = get_summary_df(prederrs_df, save_csv=true, path_to_results = path_to_results, filename="prederr_stats_allcombs_mle")