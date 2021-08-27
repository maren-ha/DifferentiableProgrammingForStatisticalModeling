using Pkg;
Pkg.activate(path_to_repo)
Pkg.instantiate()
# check that required package versions are there 
Pkg.status()

# import packages
using CSV 
using DataFrames
using FileIO
using GLM 
using JLD2
using LinearAlgebra
#using Plots
using ProgressMeter
using Random
using Statistics
using StatsBase
using StatsPlots
using VegaLite
using Zygote 

#load data  
datadict = load("data/data_allhospitals.jld2")
hospital_list = datadict["hospital_dfs"]

include("evaluate.jl")
include("loss.jl")

path_to_results = string(path_to_repo, "/results/")