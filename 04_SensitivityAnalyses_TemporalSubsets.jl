"""
Sensitivity Analysis II: Use temporal subsets of data to compare all combinations 
of global and local parameter estimation and baseline models on all subsets

Step 1: Fit the model on all subsets half as long as the entire time interval 
for all combinations of global/individual parameters in parallel and save the parameters
"""
#
# Setup 
#

# cd("path/to/Github/repo")
path_to_repo = pwd()
include("src/setup.jl")
path_to_results = string(path_to_results, "temporalsubsets/")

# run code in parallel to reduce runtime
using Distributed
addprocs(6) # add desired/available number of parallel processes to run
println(nprocs())

# now repeat the setup so that is configured on all processes
@eval @everywhere path_to_repo = $path_to_repo
@eval @everywhere path_to_results = $path_to_results
@everywhere cd(path_to_repo)

@everywhere using Pkg;
@everywhere Pkg.activate(path_to_repo)
@everywhere Pkg.instantiate()
# check that required package versions are there 
@everywhere Pkg.status()

# import packages
@everywhere using DataFrames
@everywhere using Distributed
@everywhere using FileIO
@everywhere using GLM 
@everywhere using JLD2
@everywhere using LinearAlgebra
@everywhere using ProgressMeter
@everywhere using Random
@everywhere using Statistics
@everywhere using StatsBase
@everywhere using StatsPlots
@everywhere using VegaLite
@everywhere using Zygote 

@everywhere include("src/evaluate.jl")
@everywhere include("src/loss.jl")

#load data  
datadict = load("data/data_allhospitals.jld2")
hospital_list = datadict["hospital_dfs"]
@eval @everywhere hospital_list = $hospital_list

#
# define functions to build the temporal subsets of data and run the model 
# for all combinations of global vs. individual parameter estimation
#

@everywhere function builddatafromrange(hospital_list, resamplerange)
    dfs = []
    inds_in_dfs = []
    for i = 1:length(hospital_list)
        df = hospital_list[i][resamplerange,:]
        if sum(df[1:(end-1),:notreported]) >= (nrow(df[1:(end-1),:])-1)
            continue
        else
            push!(inds_in_dfs, i)
            push!(dfs, df)
        end
    end
    return dfs, inds_in_dfs
end

@everywhere function runlocalglobalmix(hospital_list, combination::Array{String,1}, nepochs)

    nhospitals = length(hospital_list)
    betas = [[0.0; 0.0; 0.0] for i in 1:length(hospital_list)]
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]

    for epoch in 1:nepochs
        for i = 1:nhospitals
            y, z, r = hospital_list[i][1:end-1,:currentcases], hospital_list[i][1:end-1,:incidences], hospital_list[i][1:end-1,:reported]
            beta = betas[i]
            curgrad = gradient(arg -> loss(y, z, r, arg), beta)
            beta .-= [0.001, 0.01, 0.001] .* curgrad[1]
            betas[i] = beta
        end

        feasibleinds = findall(x -> (!any(isnan.(x)) && !any(abs.(x) .> 10)), betas) # exclude betas from non-converged fits

        if beta1 == "global"
            for i in feasibleinds #length(betas[feasibleinds]) # reproduces previous restuls 
                betas[i][1] = mean(collect(x[1] for x in betas[feasibleinds]))
            end
        end

        if beta2 == "global"
            for i in feasibleinds #length(betas[feasibleinds]) # reproduces previous restuls 
                betas[i][2] = mean(collect(x[2] for x in betas[feasibleinds]))
            end
        end

        if beta3 == "global"
            for i in feasibleinds #length(betas[feasibleinds]) # reproduces previous restuls 
                betas[i][3] = mean(collect(x[3] for x in betas[feasibleinds]))
            end
        end
    end
    return betas
end

#
# Now, for each combination of global/individual parameters, the model is fitted on each temporal subset 
# called "resaple" in the code and the estimated parameters as well as the hospitals where estimation 
# converged are saved in the "temporalsubsets"- subfolder of the results folder. 
#

# first, we need to define the hyperparameters of the optimization
@everywhere nepochs = 1000
@everywhere ntimepoints = length(hospital_list[1][!,:date])
@everywhere lateststart = Int(floor(ntimepoints/2))
@everywhere nsampletimepoints = lateststart

# we define all combinations of global and local coefficients
@everywhere combinations = [["global", "global", "global"],
                ["global", "global", "individual"],
                ["global", "individual", "global"],
                ["global", "individual", "individual"],
                ["individual", "global", "global"],
                ["individual", "global", "individual"],
                ["individual", "individual", "global"],
                ["individual", "individual", "individual"]
]

# and finally, we are ready to get going and run all combinations in parallel 
@sync @distributed for cind in 1:length(combinations)
    combination = combinations[cind]
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]
    @info "starting with new combination: $(beta1), $(beta2), $(beta3)..." 
    betas_allresamples = []
    inds_in_dfs_allresamples = []
    for resample in 1:(lateststart+1)
        @info "starting with resample $(resample)..."
        resamplerange = resample:resample+nsampletimepoints-1
        dfs, inds_in_dfs = builddatafromrange(hospital_list, resamplerange)
        betas = runlocalglobalmix(dfs, combination, nepochs)
        push!(betas_allresamples, betas)
        push!(inds_in_dfs_allresamples, inds_in_dfs)
        @info "betas calculated!"
    end
    save(string(path_to_results, "betas_allsubsets_combination$(cind).jld2"), Dict("betas_resamples_combination$(cind)" => betas_allresamples))
    save(string(path_to_results, "inds_in_dfs_allsubsets_combination$(cind).jld2"), Dict("inds_resamples_combination$(cind)" => inds_in_dfs_allresamples))
end