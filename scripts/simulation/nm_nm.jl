using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles
using DSP
using SpecialFunctions
using DelaySSAToolkit
using Catalyst
using StatsBase, Distributions
using JLD2
using Interp1d
using JSON
using DataFrames
using CSV
# using  Flux
function TX_model(parms::Vector{Float64}, tf::Float64)
    # 1 OFF; 2 ON, 3 nascentRNA, 4 matureRNA

    # OFF => ON                 non-Markovian reaction, Gamma distribution with shape α1 and mean μ1
    # ON => OFF                 non-Markovian reaction, Gamma distribution with shape α2 and mean μ2
    # ON --> ON + nascentRNA    Markovian reaction, geometric distribution with rate ρ
    # nascentRNA => matureRNA   non-Markovian reaction, Gamma distribution with shape α3 and mean μ3
    # matureRNA => 0            non-Markovian reaction, Gamma distribution with shape α4 and mean μ4

    # Gamma distribution referring https://juliastats.org/Distributions.jl/v0.14/univariate.html#Distributions.Gamma

    α1, μ1, α2, μ2, α3, μ3, α4, μ4, ρ  = parms

    rates = [ρ]
    reactant_stoich = [[2=>1]]
    net_stoich = [[3=>1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates=false)
    jumpset = DelaySSAToolkit.JumpSet((), (), nothing, mass_jump)

    
    u0 = [1,0,0,0]  # [OFF, ON, nascentRNA, matureRNA] initial values
    de_chan0 = [[1e-8], [], [], []]
    tspan = (0, tf)
    dprob = DiscreteProblem(u0, tspan)

    # nascentRNA => matureRNA trigger
    delay_trigger_affect1! = function (integrator, rng)
        α = α3
        μ = μ3
        τ = rand(Gamma(α, μ / α))
        append!(integrator.de_chan[3], τ)
    end

    #  OFF =>  ON complete; ON => OFF trigger
    delay_complete_affect1! = function (integrator, rng)
        integrator.u[1] -= 1 # OFF state minus 1
        integrator.u[2] += 1 # ON state plus 1
        α = α2
        μ = μ2
        τ = rand(Gamma(α, μ / α))
        append!(integrator.de_chan[2], τ) # add to the delay channel
    end

    #  ON =>  OFF complete; OFF => ON trigger
    delay_complete_affect2! = function (integrator, rng)
        integrator.u[2] -= 1 # OFF state minus 1
        integrator.u[1] += 1 # ON state plus 1
        α = α1
        μ = μ1
        τ = rand(Gamma(α, μ / α))
        append!(integrator.de_chan[1], τ) # add to the delay channel
    end

    #  nascentRNA => matureRNA complete; matureRNA => 0 trigger
    delay_complete_affect3! = function (integrator, rng)
        integrator.u[3] -= 1 # nascentRNA minus 1
        integrator.u[4] += 1 # matureRNA plus 1
        α = α4
        μ = μ4
        τ = rand(Gamma(α, μ / α))
        append!(integrator.de_chan[4], τ) # add to the delay channel
    end

    #  matureRNA => 0 complete
    delay_complete_affect4! = function (integrator, rng)
        integrator.u[4] -= 1 # matureRNA minus 1
    end

    delay_trigger = Dict(1 => delay_trigger_affect1!)
    delay_complete = Dict(1 => delay_complete_affect1!, 2 => delay_complete_affect2!, 3 => delay_complete_affect3!,
        4 => delay_complete_affect4!)
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)


    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset, de_chan0, save_positions=(true, true))


    # tt = 0:0.5:30

    # sol = solve(djprob, DelaySSAToolkit.SSAStepper(),saveat=tt)
    sol = solve(djprob, DelaySSAToolkit.SSAStepper())

    sol
end



function generate_param(min_val, max_val,data_number)
    result = min_val .+ (max_val-min_val) .*rand(data_number) 
    return(result)
end


function interp_data(yi::Vector{Int64}, xi::Vector{Float64})
    yi = convert(Vector{Float64}, yi)
    x = collect(2000:0.01:3000)
    a = Vector{Float32}()
    for mode in INTERP_MODE_LIST
        if string(mode) == "Previous"
            b_index = 1
            f = interp(xi[b_index:length(xi)], yi[b_index:length(xi)], mode) # get an interpolation function
            y = f.(x) # Do interpolation
            a = y
        end
    end
    a = convert(Vector{Int64}, a)
    a
end


function simulate(param_number::Int64)
    list_str = [Dict{String, Int64}() for _ in 1:param_number]
    list_parameter = [Vector{Float64}() for _ in 1:param_number]
    list_mean =  Vector{Matrix{Float64}}(undef, param_number)
    list_std = Vector{Matrix{Float64}}(undef, param_number)
    max_value = Vector{Float64}()

    α1_li = generate_param(0.01, 5,param_number)
    μ1_li = generate_param(0.01, 5,param_number)
    α2_li = generate_param(0.01, 5,param_number)
    μ2_li = generate_param(0.01, 5,param_number)
    α3_li = generate_param(0.01, 5,param_number)
    μ3_li = generate_param(0.01, 5,param_number)
    α4_li = generate_param(0.01, 5,param_number)
    μ4_li = generate_param(0.01, 5,param_number)
    ρ_li =  generate_param(1, 30,param_number)

    for i = 1:param_number
        print(i)
        parms = Float64[α1_li[i], μ1_li[i], α2_li[i], μ2_li[i], α3_li[i], μ3_li[i], α4_li[i], μ4_li[i], ρ_li[i]]
        result = TX_model(parms,3000.0)
        matrix_result = hcat(result.u...)
        try
            on_state = interp_data(matrix_result[1,:],result.t)
            nascentRNA = interp_data(matrix_result[3,:],result.t)
            matureRNA = interp_data(matrix_result[4,:],result.t)
            sol_u_sub = cat(on_state,nascentRNA, matureRNA, dims=2)
            list_parameter[i] = parms
            temp_str = join.(eachrow(sol_u_sub),"_")
            list_str[i] = countmap(temp_str)
            push!(max_value,findmax(sol_u_sub)[1])
            list_mean[i]= mean(sol_u_sub,dims=1)
            list_std[i] = std(sol_u_sub,dims=1)
        catch e
            print("error")
            print(e)
            continue
    end

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

# # parameter values
# α1 = 2.0
# μ1 = 1.0
# α2 = 2.0
# μ2 = 1.0
# α3 = 2.0
# μ3 = 1.0
# α4 = 2.0
# μ4 = 1.0
# ρ = 20.0

# parms = [α1, μ1, α2, μ2, α3, μ3, α4, μ4, ρ]

# result = TX_model(parms,50.0)
# matrix_result = hcat(result.u...)
# on_state = interp_data(matrix_result[1,:],result.t)
# nascentRNA = interp_data(matrix_result[3,:],result.t)
# matureRNA = interp_data(matrix_result[4,:],result.t)
# sol_u_sub = cat(on_state,nascentRNA, matureRNA, dims=2)



# train_file
model_name = "nm_nm"
root_path = "D:/codeRelate/AcademicRelateCode/nessie/af1/data"
data_size = 11000
result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate(data_size)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
open("$root_path/train_$(model_name)_stable_$max_val_ds.json", "w") do f
    write(f, json_str)
end

df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/train_$(model_name)_stable_parameter_$data_size.csv", df)
truncate_max = findmax(matrix_mean+3*matrix_std,dims=1)[1]
CSV.write("$root_path/train_$(model_name)_truncate_max_$data_size.csv",DataFrame(truncate_max, :auto))


# data_size = 1000
# result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate(data_size)
# json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)
# open("$root_path/valid_$(model_name)_stable_$max_val_ds.json", "w") do f
#     write(f, json_str)
# end

# df = DataFrame(parameters_matrix, :auto)
# CSV.write("$root_path/valid_$(model_name)_stable_parameter_$data_size.csv", df)
# truncate_max = findmax(matrix_mean+3*matrix_std,dims=1)[1]
# CSV.write("$root_path/valid_$(model_name)_truncate_max_$data_size.csv",DataFrame(truncate_max, :auto))

