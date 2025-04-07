using Catalyst
using StaticArrays
# using DifferentialEquations
using Distributions
using JumpProcesses
using CSV, DataFrames
using JSON
using StatsBase

@parameters σ_u σ_b ρ_u ρ_b
ranges = [
    0 2
    0 0.1
    0 10
    0 100
]

ranges = [
    0.9 1.1
    0.01 0.1
    0.9 1.1
    1. 25.
]

rn = @reaction_network begin
    @parameters σ_u σ_b ρ_u ρ_b
    σ_u * (1 - G), 0 --> G + P
    σ_b, G + P --> 0
    ρ_u, G --> G + P
    ρ_b * (1 - G), 0 --> P
    1, P --> 0
end 


function simulate3!(N::Int64;test_flag = false)
    # @unpack Gu_A, Pn_B,Gu_B,Pn_A = rn
    tt = 5000:0.05:10000
    tt = 10000:0.1:20000
    u0 = @SArray [ 0,1 ]
    list_str = [Dict{String, Int64}() for _ in 1:N]
    list_parameter = [Vector{Float64}() for _ in 1:N]
    max_value = Vector{Float64}()
    jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
    dprob = DiscreteProblem(jsys, u0, (0.0, last(tt)), zeros(Float64, numreactionparams(rn)))
    jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

    equidistant_params = get_equidistant_params()
    Threads.@threads for i in 1:N
        print("$i ")
        ps = rand(prior)
        if test_flag
            ps = equidistant_params[i]
        end
        print("the ps is",ps)
        list_parameter[i] = ps
        jprob = remake(jprob, tspan=(0.0, last(tt)), p=ps)
        sol = solve(jprob, SSAStepper(),saveat=tt)
        sol_u_matrix = hcat(sol.u...)'
        # G , P species(rn)
        keep_vari = [1,2]
        sol_u_sub = sol_u_matrix[:,keep_vari]
        temp_str = join.(eachrow(sol_u_sub),"_")
        list_str[i] = countmap(temp_str)
        push!(max_value,findmax(sol_u_sub)[1])
    end
    list_str,list_parameter,max_value
end


function processData(result3,parameters_vec,max_value,data_size)
    max_val = Int(findmax(max_value)[1])
    parameters_matrix = hcat(parameters_vec...)
    json_str = JSON.json(result3)
    max_val_ds = string(max_val)*"_"*string(data_size)
    json_str,parameters_matrix,max_val_ds
end

function get_equidistant_params()
    # # Set the number of pairs
    n_pairs = 100
    # # Generate equidistant values for each interval
    x_values = range(0.01, stop=0.1, length=n_pairs)
    y_values = range(1., stop=100., length=n_pairs)
    list_of_vectors = []
    # Create the pairs
    for i in 1:n_pairs
        for j in 1:n_pairs
            push!(list_of_vectors, [1.,x_values[i],1.,y_values[j]])
        end 
    end
    list_of_vectors
end

root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/gptTrain/data/afl"

prior = Product(Uniform.(ranges[:,1], ranges[:,2]))
# train_file
data_size = 20000
data_size = 5100
result3,parameters_ve,max_value = simulate3!(data_size,test_flag = false)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/train_afl_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end

df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/train_afl_stable_parameter_$data_size.csv", df)

# valid_file
data_size = 510
result3,parameters_ve,max_value = simulate3!(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/valid_afl_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end
df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/valid_afl_stable_parameter_$data_size.csv", df)

# # test_file
# data_size = 10000
# result3,parameters_ve,max_value = simulate3!(data_size,test_flag = true)
# json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
# open("$root_path/test_afl_stable_$max_val_ds.json", "w") do f
#     write(f, json_str)
# end
# df = DataFrame(parameters_matrix, :auto)
# CSV.write("$root_path/test_afl_stable_parameter_$data_size.csv", df)

