"""
Sensitivity Analysis II: Recovering randomly censored values from all hospitals with no missing reports
"""

#
# Setup  
#

# cd("path/to/Github/repo")
path_to_repo = pwd()
include("src/setup.jl")
include("src/lov.jl")

#
# First, identify densely sampled hospitals: find all hospitals where no observations are missing 
#

nhospitals = length(hospital_list)
allreported_inds = findall(i -> sum(hospital_list[i][!,:notreported])== 0, 1:nhospitals)

#
# Scenario 1: for those, randomly remove 10, 25 and 50 percent of values 
#

# initialize array to save prediction errors
prederrs_df_all = []

# fit prediction models (all local coefficients) 
for frac_missing in [0.1, 0.25, 0.5, 0.75] 

    @info "Starting scenario with $(frac_missing) percent random censoring"

    betas = collect([0.0; 0.0; 0.0] for ind in 1:length(allreported_inds));
    lrbetas = collect([0.0; 0.0; 0.0] for ind in 1:length(allreported_inds));

    prederrs_mean = []
    prederrs_modmean = []
    prederrs_zero = []
    prederrs_lr = []
    prederrs_incr = []

    Random.seed!(42);
    nreps = 100

    @showprogress for ind in 1:length(allreported_inds)
        i = allreported_inds[ind]
        true_y, z, r = hospital_list[i][:,:currentcases], hospital_list[i][:,:incidences], hospital_list[i][:,:reported]
        
        prederr_incr = 0.0
        prederr_lr = 0.0
        prederr_zero = 0.0
        prederr_mean = 0.0
        prederr_modmean = 0.0
        
        mod_r = modify_r_wrapper(frac_missing)

        prederr_mean, prederr_modmean, prederr_zero, prederr_lr, prederr_incr, betas_reps, lrbetas_reps = fitmodels_lov(hospital_list, i, nreps, mod_r)

        betas[ind] = [mean(collect(betas_reps[rep][1] for rep in 1:nreps)); mean(collect(betas_reps[rep][2] for rep in 1:nreps)); mean(collect(betas_reps[rep][3] for rep in 1:nreps))]
        lrbetas[ind] = [mean(collect(lrbetas_reps[rep][1] for rep in 1:nreps)); mean(collect(lrbetas_reps[rep][2] for rep in 1:nreps)); mean(collect(lrbetas_reps[rep][3] for rep in 1:nreps))]
        
        push!(prederrs_mean, prederr_mean/nreps)
        push!(prederrs_modmean, prederr_modmean/nreps)
        push!(prederrs_zero, prederr_zero/nreps)
        push!(prederrs_lr, prederr_lr/nreps)
        push!(prederrs_incr, prederr_incr/nreps)
    end

    prederrs_df = create_prederrs_df(prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_lr, prederrs_incr)
    push!(prederrs_df_all, prederrs_df)
end

# save results 
path_to_results = string(path_to_repo, "/results/")
save(string(path_to_results, "randomcensoring_prederrs.jld2"), Dict("all_prederrs_dfs" => prederrs_df_all))

# if you have done the analyses before, load and create boxplots and summary statistics
prederrs_dict = load(string(path_to_results, "randomcensoring_prederrs.jld2"))
prederrs_df_all = prederrs_dict["all_prederrs_dfs"]

# create dataframe of summary statistics 
nmodels = 5
nscenarios= 4
mses = []
medians = []
firstquartiles = []
thirdquartiles = []
maximums = []
for i in 1:nscenarios 
    cur_pred_df = prederrs_df_all[i]
    push!(mses, round.(sum.(eachcol(cur_pred_df[!,2:end]))/nrow(cur_pred_df), digits = 3))
    push!(medians,round.(median.(eachcol(cur_pred_df[!,2:end])), digits = 3))
    push!(firstquartiles, round.(quantile.(eachcol(cur_pred_df[!,2:end]), 0.25), digits = 3))
    push!(thirdquartiles, round.(quantile.(eachcol(cur_pred_df[!,2:end]), 0.75), digits = 3))
    push!(maximums, round.(maximum.(eachcol(cur_pred_df[!,2:end])), digits=3))
end

summary_df = DataFrame(Scenario = repeat(["$(i)" for i in 1:nscenarios], inner=nmodels),
                    Model = repeat(["Mean model", "Modified mean model", "Zero model", "Linear regression", "Increment model"], nscenarios),
                    MSE = cat(mses..., dims=1),
                    FirstQuartile = cat(firstquartiles..., dims=1),
                    Median = cat(medians..., dims=1),
                    ThirdQuartile = cat(thirdquartiles..., dims=1),
                    Maximum = cat(maximums..., dims=1)
)
CSV.write(string(path_to_results, "randomcensoring_summary.csv"), summary_df)