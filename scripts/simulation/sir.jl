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
#     ρ, 0--> S
#     k_1, S + I --> 2I
#     k_2, I --> R
#     k_3, R --> S
#     δ_s, S --> 0
#     δ_i, I --> 0
#     δ_r, R --> 0
# end 


ranges = [
1.0   20.0 # ρ
0.1   0.2  # k_1
0.1   0.2  # k_2
0.1   0.2  # k_3
0.1   0.2  # δ_s
0.1   0.2  # δ_i 
0.1   0.2  # δ_r  
]

rn = @reaction_network begin
    @parameters ρ_1 ρ_2 ρ_3 r k_1 k_2 k_3 δ_s δ_i δ_r
    ρ_1, 0--> S
    ρ_2, 0--> I
    ρ_3, 0--> R
    r, S--> 2S
    k_1, S + I --> 2I
    k_2, I --> R
    k_3, R --> S
    δ_s, S --> 0
    δ_i, I --> 0
    δ_r, R --> 0
end 

ranges = [
1.0   10.0 # ρ_1
1.0   10.0 # ρ_2
1.0   10.0 # ρ_3
0.02  0.05  # r
0.02   0.1  # k_1
0.1   0.2  # k_2
# 0.1   0.3  # k_3
0.2   0.4  # k_3
0.1   0.2  # δ_s
0.1   0.2  # δ_i 
0.1   0.2  # δ_r  
] 


function simulate3!(N::Int64)
    # @unpack M_A,P_A = rn
    # G_uA ,G_bA,M_A,P_A species(rn)
    keep_vari = [1,2,3]
    # tt = 5000:0.05:10000
    # tt = 2000:0.02:3000 # v1
    tt = 2000:0.01:3000 # 
    tt = 20000:0.01:30000 # 

    u0 = @SArray [ 1,2,0 ]
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

model_name = "sir"
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/gptTrain/data/$model_name"
prior = Product(Uniform.(ranges[:,1], ranges[:,2]))

# train_file
data_size = 16000
data_size = 1100
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
data_size = 1600
data_size = 110
result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate3!(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/valid_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end

df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/valid_$(model_name)_stable_parameter_$data_size.csv", df)
/GPUFS/sysu_jjzhang_1/.conda/envs/hzwMetCopy/bin/python /GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/gptTrain/trainer2.py