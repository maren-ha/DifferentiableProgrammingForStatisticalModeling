"""
Sensitivity Analysis II: Use temporal subsets of data to compare all combinations 
of global and local parameter estimation and baseline models on all subsets

Step 2: evaluate the results based on the estimated parameters from step 1
"""
#
# Setup 
#

# cd("path/to/Github/repo")
path_to_repo = pwd()
include("src/setup.jl")
include("src/lov.jl")

#
# Function: build dataset from a range of the time interval 
#

function builddatafromrange(hospital_list, resamplerange)
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
 
#
# Evaluate: build dataframe with summary statistics: mean, median, diffs between models for each combination 
# Table 2 and Supplementary Tables 3, 4 and 5. 
#

# first, define the combinations of global/individual estimation of parameters
combinations = [["global", "global", "global"],
                ["global", "global", "individual"],
                ["global", "individual", "global"],
                ["global", "individual", "individual"],
                ["individual", "global", "global"],
                ["individual", "global", "individual"],
                ["individual", "individual", "global"],
                ["individual", "individual", "individual"]
]

# some definitions
ntimepoints = length(hospital_list[1][!,:date])
lateststart = Int(floor(ntimepoints/2))
nsampletimepoints = lateststart

# initialize an empty dictionary to save the data from all combinations into 
prederrs_resamples_dict = Dict()
# now, go through the combinations of global and individual parameters, load the corresponding data, evaluate the prediction errors and combine
for cind in 1:length(combinations)

    combination = combinations[cind]
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]
    @info "starting with new combination: $(beta1), $(beta2), $(beta3)..." 
    # load data
    betas_dict = load(string(path_to_results, "temporalsubsets/betas_allsubsets_combination$(cind).jld2"))
    betas_allresamples = betas_dict["betas_resamples_combination$(cind)"];
    inds_dict = load(string(path_to_results, "temporalsubsets/inds_in_dfs_allsubsets_combination$(cind).jld2"))
    inds_in_dfs_allresamples = inds_dict["inds_resamples_combination$(cind)"];
    # get prediction errors 
    prederrs_dfs_resamples = []
    for resample in 1:(lateststart+1)
        betas = betas_allresamples[resample]
        hospitalinds = inds_in_dfs_allresamples[resample]

        resamplerange = resample:resample+nsampletimepoints-1
        dfs, inds_in_dfs = builddatafromrange(hospital_list, resamplerange)

        prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod = evaluate_predictions(dfs, betas)
        prederrs_df = create_prederrs_df(prederrs_mean, prederrs_modmean, prederrs_zero, prederrs_linreg, prederrs_incrmod, hospitalinds)
        push!(prederrs_dfs_resamples, prederrs_df)
    end
    # combine into overall dictionary
    prederrs_resamples_dict["prederrs_dfs_combination$(cind)"] = prederrs_dfs_resamples
end

# next, aggregate the per-hospital prediction errors, calculating the summary statistics reported in the manuscript 
agg_prederrs_dict = Dict()
for cind in 1:length(combinations)
    prederrs_dfs_resamples = prederrs_resamples_dict["prederrs_dfs_combination$(cind)"]

    agg_prederrs_resamples = DataFrame()

    for resample in 1:(lateststart+1)
        for colname in names(prederrs_dfs_resamples[resample])[2:end]
            if resample == 1 
                agg_prederrs_resamples[!,string(colname, "_sse")] = fill(0.0, length(prederrs_dfs_resamples))
                agg_prederrs_resamples[!,string(colname, "_mse")] = fill(0.0, length(prederrs_dfs_resamples))
                agg_prederrs_resamples[!,string(colname, "_median")] = fill(0.0, length(prederrs_dfs_resamples))
            end
            agg_prederrs_resamples[resample, string(colname, "_sse")] = sum(prederrs_dfs_resamples[resample][:,colname])#/nrow(prederrs_dfs_resamples[resample])
            agg_prederrs_resamples[resample, string(colname, "_mse")] = sum(prederrs_dfs_resamples[resample][:,colname])/nrow(prederrs_dfs_resamples[resample])
            agg_prederrs_resamples[resample, string(colname, "_median")] = median(prederrs_dfs_resamples[resample][:,colname])
            if resample == (lateststart+1)
                for colname in names(prederrs_dfs_resamples[resample])[2:(end-1)]
                    agg_prederrs_resamples[!, string("sse_diffs_to_", colname)] = agg_prederrs_resamples[:,string(colname, "_sse")] .- agg_prederrs_resamples[:,"incrmodel_sse"]
                    agg_prederrs_resamples[!, string("mse_diffs_to_", colname)] = agg_prederrs_resamples[:,string(colname, "_mse")] .- agg_prederrs_resamples[:,"incrmodel_mse"]
                end
            end
        end
    end
    CSV.write(string(path_to_results, "agg_prederrs_subsets_combination$(cind).csv"), agg_prederrs_resamples)
    agg_prederrs_dict["combination$(cind)"] = agg_prederrs_resamples
end

# calculate the relative improvements of the increment model over each baseline model and calculate quantiles
allcombinations_quantiles_df = DataFrame()
allcombinations_quantiles_df[!,"combination"] = ["global, global, global", "global, global, individual", 
    "global, individual, global", "global, individual, individual", "individual, global, global",
    "individual, global, individual", "individual, individual, global", "individual, individual, individual"]
for curquantile in [0.25, 0.5, 0.75]
    allcombinations_quantiles_df[!,string("meanmodel_", curquantile)] = fill(0.0, length(combinations))
    allcombinations_quantiles_df[!,string("modmeanmodel_", curquantile)] = fill(0.0, length(combinations))
    allcombinations_quantiles_df[!,string("zeromodel_", curquantile)] = fill(0.0, length(combinations))
    allcombinations_quantiles_df[!,string("linregmodel_", curquantile)] = fill(0.0, length(combinations))
end
# make a boxplot for each combination 
for cind in 1:length(combinations)
    agg_prederrs_resamples = agg_prederrs_dict["combination$(cind)"]
    diff_cols = names(agg_prederrs_resamples)[findall(x -> occursin("sse_diffs_to", x), names(agg_prederrs_resamples))]

    for col in diff_cols 
        for curquantile in [0.25, 0.5, 0.75]
            modelname = split(col, "_")[4]
            newcolname = string(modelname, "_$(curquantile)")
            allcombinations_quantiles_df[cind,newcolname] = round(quantile(agg_prederrs_resamples[!,col], curquantile),digits = 3)
        end
    end

    stacked_prederrs=stack(agg_prederrs_resamples[!,diff_cols])
    rename!(stacked_prederrs, ["model", "prediction_error"])
    stacked_prederrs |>
    @vlplot(
        width=400, height=500,
        mark={:boxplot, extend=1.5},
        x={"model:n", title= ""},#, axis = {values=[1,2,3,4,5]}, scale ={domain=["meanmodel", "modmeanmodel", "zeromodel", "linregmodel", "incrmodel"]}},
        y={"prediction_error", title="prediction error"},
        color=:model,
        size={value=50},
        title = "Differences in prediction error for combination $(cind)"
    ) |> save(string(path_to_results, "prederr_diffs_boxplots_combination_$(cind).svg"))
end
# save to results folder
CSV.write(string(path_to_results, "prederrs_allcombinations_quantiles.csv"), allcombinations_quantiles_df)


#
# Evaluate: make overlayed plots of resamples for exemplary hospital (Figure 2) 
#

nresamples = lateststart +1
i=40
df = hospital_list[i]
y, z, r = df[1:end-1,:currentcases], df[1:end-1,:incidences], df[1:end-1,:reported]

# get betas for each combination, first from the estimation based on the complete data 
betas_all_combs_dict = load(string(path_to_results, "betas_allcombinations.jld2"))
betas_all_combs = betas_all_combs_dict["betas_allcombinations"]

allbetas = []
betas_alldata = []
for cind in 1:length(combinations)
    combination = combinations[cind]
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]

    push!(betas_alldata, betas_all_combs["combination_$(beta1)_$(beta2)_$(beta3)"][i])

    inds_in_dfs_resamples_dict = load(string(path_to_results, "inds_in_dfs_allresamples_combination$(cind).jld2"))
    inds_in_dfs_allresamples = inds_in_dfs_resamples_dict["inds_resamples_combination$(cind)"]

    betas_dict = load(string(path_to_results, "betas_allresamples_combination$(cind).jld2"))
    betas_allresamples = betas_dict["betas_resamples_combination$(cind)"];
    push!(allbetas, collect(betas_allresamples[resample][findall(x -> x == 40,inds_in_dfs_allresamples[resample])[1]] for resample in 1:nresamples))
end

# for each combination, make a plot with all resamples overlayed 
for cind in 1:length(combinations)
    @info "starting with combination $(cind)"
    combination = combinations[cind]
    beta1, beta2, beta3 = combination[1], combination[2], combination[3]

    beta_alldata = betas_alldata[cind]
    pred_traj = getpredictedtrajectory(y, z, r, beta_alldata)
    pred_df_all = DataFrame(date =  df[findfirst(x -> x, df[:,:reported])+1:(end-1),:date],
                    predicted_cases = pred_traj,
                    notreported = df[findfirst(x -> x, df[:,:reported])+1:(end-1),:notreported]
    )
    p = pred_df_all |> @vlplot(width = 1000, height = 300, 
                    transform = [{calculate = "datum.notreported == true ? 'predicted' : 'reported'",
                                as = "casesreported"}]) 
    @vlplot(
            mark = {:point, filled=true, size=175, opacity=0.75},
            x = {:date, title = "Date"},
            y = {"predicted_cases:q", title = "Number of prevalent COVID-19 cases"},
            color = {"casesreported:n",
                    scale = {range = ["#54a24b", "#4c78a8"]},
                    legend = {title = "Prevalent cases reported?"}
                    }
        ) +
        @vlplot(
            mark = {:line, color = "#4c78a8", opacity = 0.75},
            x = {"date:t", title = "Date"},
            y = {"predicted_cases:q", title = "Number of prevalent COVID-19 cases"}
    )
    for resample in 1:nresamples 
        resamplerange = resample:resample+nsampletimepoints-1
        resample_df = df[resamplerange,:]
        cur_beta = allbetas[cind][resample]
        cur_y, cur_z, cur_r = resample_df[1:end-1,:currentcases], resample_df[1:end-1,:incidences], resample_df[1:end-1,:reported]
        pred_traj = getpredictedtrajectory(cur_y, cur_z, cur_r, cur_beta)
        pred_df = DataFrame(date =  resample_df[findfirst(x -> x, resample_df[:,:reported])+1:(end-1),:date],
                            predicted_cases = pred_traj,
                            notreported = resample_df[findfirst(x -> x, resample_df[:,:reported])+1:(end-1),:notreported]
        )
        p = p + @vlplot(data = pred_df[pred_df[!,:notreported].== true,:],
            mark = {:point, filled=true, size=150, opacity=0.5, color = "#b2df8a"},
            x = {:date, title = "Date"},
            y = {"predicted_cases:q", title = "Number of prevalent COVID-19 cases"}
        )
    end
    p = p + @vlplot(
        data=pred_df_all,
        transform = [{calculate = "datum.notreported == true ? 'predicted' : 'reported'",
                                as = "casesreported"}],
        mark = {:point, filled=true, size=175, opacity=0.75},
        x = {:date, title = "Date"},
        y = {"predicted_cases:q", title = "Number of prevalent COVID-19 cases"},
        color = {"casesreported:n",
            scale = {range = ["#54a24b", "#4c78a8"]},
            legend = {title = "Prevalent cases reported?"}
            }
        ) +
    @vlplot(data=pred_df_all,
        title = "$(beta1) intercept, $(beta2) current cases, $(beta3) predicted incidences",
        mark = {:line, color = "#4c78a8", opacity = 0.75},
        x = {"date:t", title = "Date"},
        y = {"predicted_cases:q", title = "Number of prevalent COVID-19 cases"}
    )
    display(p)
    p |> save(string(path_to_results,"predicted_cases_hospital$(i)_combination_$(cind).svg"))
end
