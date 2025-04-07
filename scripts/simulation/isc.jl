using Catalyst
using StaticArrays
# using DifferentialEquations
using Distributions
using JumpProcesses
using CSV, DataFrames
using JSON
using StatsBase

# rn = @reaction_network begin
#     @parameters ρ k_1 k_2 k_3 δ_s δ_i δ_r
#     ρ, 0 --> X1
#     r_1, X1 --> X2
#     r_2, X2 --> X3
#     r_3, X3 --> X4
#     r_4, X4 --> X5
#     r_5, X5 --> X6
#     r_6, X6 --> X7
#     r_7, X7 --> X8
#     r_8, X8 --> X9
#     r_9, X9 --> X10
#     δ_1, X1 --> 0
#     δ_2, X2 --> 0
#     δ_3, X3 --> 0
#     δ_4, X4 --> 0
#     δ_5, X5 --> 0
#     δ_6, X6 --> 0
#     δ_7, X7 --> 0
#     δ_8, X8 --> 0
#     δ_9, X9 --> 0
#     δ_10, X10 --> 0
# end 

rn = @reaction_network begin
    @parameters ρ r_1 δ_1
    ρ, 0 --> X1
    r_1, X1 --> X2
    r_1, X2 --> X3
    r_1, X3 --> X4
    r_1, X4 --> X5
    r_1, X5 --> X6
    r_1, X6 --> X7
    r_1, X7 --> X8
    r_1, X8 --> X9
    r_1, X9 --> X10
    δ_1, X1 --> 0
    δ_1, X2 --> 0
    δ_1, X3 --> 0
    δ_1, X4 --> 0
    δ_1, X5 --> 0
    δ_1, X6 --> 0
    δ_1, X7 --> 0
    δ_1, X8 --> 0
    δ_1, X9 --> 0
    δ_1, X10 --> 0
end 
ranges = [
0.1   20 # ρ
0.1   1  # r_1
0.1   1  #δ_1
]  #v1

ranges = [
0.1   50 # ρ
0.1   1  # r_1
0.1   1  #δ_1
]  #v1


function simulate3!(N::Int64)
    # @unpack M_A,P_A = rn
    # G_uA ,G_bA,M_A,P_A species(rn)
    keep_vari = [1,2,3,4,5,6,7,8,9,10]
    # tt = 5000:0.05:10000
    tt = 1000:0.05:2000
    u0 = zeros(Int, numspecies(rn))
    u0[1] = 1
    list_str = [Dict{String, Int64}() for _ in 1:N]
    list_parameter = [Vector{Float64}() for _ in 1:N]
    list_mean =  Vector{Matrix{Float64}}(undef, N)
    list_std = Vector{Matrix{Float64}}(undef, N)
    max_value = Vector{Float64}()
    jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
    dprob = DiscreteProblem(jsys, u0, (0.0, last(tt)), zeros(Float64, numreactionparams(rn)))
    jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))
    Threads.@threads for i in 1:N
        print(i)
        ps = rand(prior)
        list_parameter[i] = ps
        jprob = remake(jprob, tspan=(0.0, last(tt)), p=ps)
        sol = solve(jprob, SSAStepper(),saveat=tt)
        sol_u_matrix = hcat(sol.u...)'
        sol_u_sub = sol_u_matrix[:,keep_vari]
        temp_str = join.(eachrow(sol_u_sub),"_")
        list_str[i] = countmap(temp_str)
        push!(max_value,findmax(sol_u_sub)[1])
        list_mean[i]= mean(sol_u_sub,dims=1)
        list_std[i] = std(sol_u_sub,dims=1)
    end
    matrix_mean = vcat(list_mean...)
    matrix_std = vcat(list_std...)
    list_str,list_parameter,max_value,matrix_mean,matrix_std
end

function processData(result3,parameters_vec,max_value,data_size)
    max_val = Int(findmax(max_value)[1])
    parameters_matrix = hcat(parameters_vec...)
    json_str = JSON.json(result3)
    max_val_ds = string(max_val)*"_"*string(data_size)
    json_str,parameters_matrix,max_val_ds
end

model_name = "isc"
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/gptTrain/data/$model_name"
prior = Product(Uniform.(ranges[:,1], ranges[:,2]))

# train_file
data_size = 15000
result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate3!(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/train_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end

df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/train_$(model_name)_stable_parameter_$data_size.csv", df)
truncate_max = findmax(matrix_mean+3*matrix_std,dims=1)[1]
CSV.write("$root_path/train_$(model_name)_truncate_max_$data_size.csv",DataFrame(truncate_max, :auto))

# valid_file
data_size = 1500
result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate3!(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/valid_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end
df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/valid_$(model_name)_stable_parameter_$data_size.csv", df)
