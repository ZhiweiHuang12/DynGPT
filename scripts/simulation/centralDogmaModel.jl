using Catalyst
using StaticArrays
# using DifferentialEquations
using Distributions
using JumpProcesses
using CSV, DataFrames
using JSON
using StatsBase
using Sobol

model_name = "ts_txl"

ranges = [
0.01   0.1  # k_off
0.01   0.1     # k_on
0.01   10     # k_synON
0.01   10     # k_synOFF
0.01   2      # k_synM 
0.2   0.5      # k_degM  
0.2   0.4       # k_degP  
]

ranges = [
0.1   2.0  # k_off
0.1   2.0     # k_on
0.01   10     # k_synON
0.01   10     # k_synOFF
0.01   2      # k_synM 
0.2   0.5      # k_degM  
0.2   0.4       # k_degP  
]

rn = @reaction_network begin
    @parameters k_off k_on k_synON k_synOFF k_synM k_degM k_degP
    (k_off, k_on), ON <--> OFF
    k_synON, ON --> ON + M
    k_synOFF, OFF --> OFF + M
    k_synM, M --> M + P
    k_degM, M --> 0
    k_degP, P --> 0
end 


function generate_params(N::Int64)
    ranges = [
        0.01   0.1  # k_off
        0.01   0.1     # k_on
        0.01   10     # k_synON
        0.01   10     # k_synOFF
        0.01   2      # k_synM 
        0.2   0.5      # k_degM  
        0.2   0.4       # k_degP  
        ]
        
    # Draw training, validation and test parameters from a Sobol sequence
    s = SobolSeq(ranges[:, 1], ranges[:, 2])
    
    params = [Sobol.next!(s) for i = 1:N]
    return(params)
end

function simulate3!(N::Int64)
    # @unpack M_A,P_A = rn
    # G_uA ,G_bA,M_A,P_A species(rn)
    keep_vari = [1,3,4]
    # tt = 5000:0.05:10000
    tt = 5000:0.05:6000
    u0 = @SArray [ 1,0,0,0 ]
    list_str = [Dict{String, Int64}() for _ in 1:N]
    list_parameter = [Vector{Float64}() for _ in 1:N]
    list_mean =  Vector{Matrix{Float64}}(undef, N)
    list_std = Vector{Matrix{Float64}}(undef, N)
    max_value = Vector{Float64}()
    jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
    dprob = DiscreteProblem(jsys, u0, (0.0, last(tt)), zeros(Float64, numreactionparams(rn)))
    jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

    params = generate_params(N)
    Threads.@threads for i in 1:N
        print("$i ")
        # ps = rand(prior)
        ps = params[i]
        list_parameter[i] = params[i]
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
    # result3_matrix = hcat(result3...)
    # counts = countmap.(eachcol(result3_matrix))
    # json_str = JSON.json(counts)
    json_str = JSON.json(result3)
    max_val_ds = string(max_val)*"_"*string(data_size)
    json_str,parameters_matrix,max_val_ds
end


root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/data/ts_txl/"
# data_size = 100
prior = Product(Uniform.(ranges[:,1], ranges[:,2]))

# train_file
data_size = 10000
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
data_size = 1100
result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate3!(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/valid_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end
df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/valid_$(model_name)_stable_parameter_$data_size.csv", df)


