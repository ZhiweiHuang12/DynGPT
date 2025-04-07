using Distributions
using DataFrames
# using Roots
using StaticArrays
using Plots

struct SSAArgs{X,Ftype,N,P}
    x0::X
    F::Ftype
    nu::N
    parms::P
    tf::Float64
end


struct SSAResult
    time::Vector{Float64}
    data::Matrix{Int64}
    args::SSAArgs
end


function pfsample(w::AbstractArray{Float64,1},s::Float64,n::Int64)
    t = rand() * s
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end


function gillespie(x0::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parms::AbstractVector{Float64},tf::Float64)
    # Args
    args = SSAArgs(x0,F,nu,parms,tf)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(x0)
    x = copy(x0')
    xa = copy(Array(x0))
    # Number of propensity functions
    numpf = size(nu,1)
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        pf = F(x,parms)
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        dt = rand(Exponential(1/sumpf))
        t += dt
        push!(ta,t)
        # Update event
        ev = pfsample(pf,sumpf,numpf)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update nsteps
        nsteps += 1
    end
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,args)
end




function F_dd(x,parms)
    (OFF1,ON1,P1,OFF2,ON2,P2) = x
    (kon1,koff1,ksyn1,kdeg1,kon2,koff2,ksyn2,kdeg2) = parms
    K = 0.6
    [kon1*OFF1/(K^2+P2^2),kon2*OFF2/(K^2+P1^2),koff1*ON1,koff2*ON2,ksyn1*ON1,ksyn2*ON2,kdeg1*P1,kdeg2*P2]
end

x0 = [1,0,0,1,0,0]
nu = [[-1 1 0 0 0 0]; [0 0 0 -1 1 0];[1 -1 0 0 0 0]; [0 0 0 1 -1 0]; [0 0 1 0 0 0];[0 0 0 0 0 1];[0 0 -1 0 0 0];[0 0 0 0 0 -1]]
parms = [5.0,0.1,100.0,0.01,5.0,0.1,50.0,0.01]
tf = 5000.0

@time result = gillespie(x0,F_dd,nu,parms,tf)
using StatsBase

counts =  result.data[2:end,6]
h_result = fit(Histogram,counts,weights(diff(result.time)),nbins=findmax(counts)[1])

plot(result.time[1:end], result.data[1:end,6], linetype=:steppre, legend = false)
histogram(h_result)
# data = ssa_data(result)
histogram(result.data[1:end-1,3] .- 0.5)
mean(result.data[1:end-1,6])
histogram(result.data[1:end-1,6] .- 0.5, bins=-0.5:1:40, weights=diff(result.time),norm=true,
          size=(600*0.9,400*0.9), grid=false, ticks=true, fontfamily="Helvetica",
          framestyle=:box, xlim=(0,30), legend=false)
xlabel!("mRNA")
ylabel!("Prob.")

dt = time()
# savefig("fig_prob_$dt.svg") 