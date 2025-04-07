using Catalyst
using StaticArrays
# using DifferentialEquations
using Distributions
using JumpProcesses
using CSV, DataFrames
using JSON
using StatsBase
using Sobol
@parameters k_on K n k_off k_syn k_deg

rn = @reaction_network begin
    @parameters k_on K n k_off k_syn k_deg 
    hillr(M,k_on,K,n)+0.1, OFF --> ON
    k_off, ON --> OFF
    k_syn, ON --> ON + M
    k_deg, M --> 0
end

function generate_params(N::Int64)
    ranges = [
        0.1 3.
        1. 5.
        -2.  2.
        0.1 3.
        0.5 40.
        0.1 1.
    ]

    ranges = [
        0.1 3.
        1. 10.
        -2.  2.
        0.1 3.
        0.5 40.
        0.1 1.
    ]

    # Draw training, validation and test parameters from a Sobol sequence
    s = SobolSeq(ranges[:, 1], ranges[:, 2])
    
    params = [Sobol.next!(s) for i = 1:N]
    return(params)
end

function simulate3!(N::Int64,;test_flag = false)
    # @unpack Gu_A, Pn_B,Gu_B,Pn_A = rn
    tt = 10000:0.1:20000
    u0 = @SArray [ 0,1,0 ]
    params = generate_params(N)
    list_str = [Dict{String, Int64}() for _ in 1:N]
    list_parameter = [Vector{Float64}() for _ in 1:N]
    max_value = Vector{Float64}()
    jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
    dprob = DiscreteProblem(jsys, u0, (0.0, last(tt)), zeros(Float64, numreactionparams(rn)))
    jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

    equidistant_params = get_equidistant_params()
    Threads.@threads for i in 1:N
        print("$i ")
        ps = params[i]
        if test_flag
            ps = equidistant_params[i]
        end
        # print("the ps is",ps)
        list_parameter[i] = ps
        jprob = remake(jprob, tspan=(0.0, last(tt)), p=ps)
        sol = solve(jprob, SSAStepper(),saveat=tt)
        sol_u_matrix = hcat(sol.u...)'
        keep_vari = [2,3]
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

model_name = "arl"
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/gptTrain/data/$model_name"
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/data/$model_name"
# prior = Product(Uniform.(ranges[:,1], ranges[:,2]))


# train_file
data_size = 60000
# data_size = 910

result3,parameters_ve,max_value = simulate3!(data_size,test_flag = false)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/train_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end
df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/train_$(model_name)_stable_parameter_$data_size.csv", df)

# valid_file
data_size = 2000
result3,parameters_ve,max_value = simulate3!(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/valid_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end
df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/valid_$(model_name)_stable_parameter_$data_size.csv", df)


