using Catalyst
using StaticArrays
using Distributions
using JumpProcesses
using CSV, DataFrames
using JSON
using StatsBase
using Sobol
# using DifferentialEquations

function generate_reaction_strings(config)
  states = config["state"]
  events = config["events"]
  reactions = []
  for event in events
      rate_expression = event["rate_expression"]
      state_change = event["state_change"]
      reactants = []
      products = []
      for (state, change) in state_change
          if change < 0
              append!(reactants, repeat([state], abs(change)))
          elseif change > 0
              append!(products, repeat([state], change))
          elseif change == 0
              # If the change is 0, add the state to both sides of the reaction
              push!(reactants, state)
              push!(products, state)
          end
      end
      reactants_str = isempty(reactants) ? "0" : join(reactants, " + ")
      products_str = isempty(products) ? "0" : join(products, " + ")
      push!(reactions, """$rate_expression, $reactants_str --> $products_str""")
  end
    return reactions
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

function simulate!(N::Int64,config_file) #::Dict{String, Any}
    # @unpack M_A,P_A = rn
    # G_uA ,G_bA,M_A,P_A species(rn)
    variable_index = config_file["variable_index"]
    # tt = 5000:0.05:10000
    t_start = config_file["t_end"]-1000
    t_end = config_file["t_end"]
    println("t_start is")
    println(t_start)
    tt = t_start:0.05:t_end
    u0 = config_file["initial_val"]
    list_str = [Dict{String, Int64}() for _ in 1:N]
    list_parameter = [Vector{Float64}() for _ in 1:N]
    list_mean =  Vector{Matrix{Float64}}(undef, N)
    list_std = Vector{Matrix{Float64}}(undef, N)
    max_value = Vector{Float64}()
    jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
    dprob = DiscreteProblem(jsys, u0, (0.0, last(tt)), zeros(Float64, numreactionparams(rn)))
    jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

    lower_bound_val = config_file["parameters"]["lower_bound_val"]
    upper_bound_val = config_file["parameters"]["upper_bound_val"]
    s = SobolSeq(lower_bound_val,upper_bound_val)
    params = [Sobol.next!(s) for i = 1:N]
    # params = generate_params(N)
    Threads.@threads for i in 1:N
        print("$i ")
        # ps = rand(prior)
        ps = params[i]
        list_parameter[i] = params[i]
        jprob = remake(jprob, tspan=(0.0, last(tt)), p=ps)
        sol = solve(jprob, SSAStepper(),saveat=tt)
        sol_u_matrix = hcat(sol.u...)'
        sol_u_sub = sol_u_matrix[:,variable_index]
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
function read_config(file_path)
  return JSON.parsefile(file_path)
end



config = JSON.parse(config_file)

config_path = ARGS[1]
config = read_config(file_path)

reactions = generate_reaction_strings(config)

# Merge reaction into a string
reaction_network_str = join(reactions, "\n    ")
# parameters_str = "kon koff ksynon ksynoff konM  kdegM kdegP"
parameters_str = join(config["parameters"]["param_names"], " ")


# Embed the reaction network into the @reaction_network macro
rn_code = """
rn = @reaction_network begin
    @parameters $parameters_str
    $reaction_network_str
end
"""
println(rn_code)
# Execute the code to generate and compile the reaction network

rn = eval(Meta.parse(rn_code))
# println(rn)

root_path = config["data_storage_folder"]
# train_file
data_size = 100
println(config)
result3,parameters_ve,max_value,matrix_mean,matrix_std = simulate!(config["train_data_size"],config)
json_str,parameters_matrix,max_val_ds = processData(result3,parameters_ve,max_value,data_size)

open("$root_path/train_$(model_name).json", "w") do f
  write(f, json_str)
end
df = DataFrame(parameters_matrix, :auto)
CSV.write("$root_path/train_$(model_name)_stable_parameter_$data_size.csv", df)
truncate_max = findmax(matrix_mean+3*matrix_std,dims=1)[1]
CSV.write("$root_path/train_$(model_name)_truncate_max_$data_size.csv",DataFrame(truncate_max, :auto))

