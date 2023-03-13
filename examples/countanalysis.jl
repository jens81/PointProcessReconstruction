## Remember to run Pkg> activate .
using Meshes, Plots, StatsBase, GLM, Combinatorics
import Random
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

import ColorSchemes.okabe_ito
cscheme = okabe_ito

#########################################
# Set up window
#########################################
W = Box((0.,0.),(1.,1.))
Wplus = enlarge(W,0.1)

#########################################
# Set up Area interaction process
#########################################
# Parameters
#R = 0.025;
R = 0.05
η = 2.7 # η = γ^(π*R^2)
λ₀ = 1000; β = λ₀/η
# Interaction function
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
# Set up exponential model (parameters and covariates)
θ₀ = [log(β),log(η)]
S = [(u,s)->1, (u,s)->-C(u,s)]
# Papangelou conditional intensity
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
# Define point Gibbs process and sample one pattern
p = GibbsProcess(logλ)

#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 100; g₀ = 1.25;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->min(sat,t(u,s))]
# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)


#########################################
# Generate original, and reconstructed patterns
#########################################
# Resolutions to be used in reconstruction
resolutions = [1,2,4,8,16,32,64]

noriginals = 500
nresolutions = length(resolutions)

patterns = [sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=30_000) for _ in 1:noriginals]
L = mean(N(s,W) for s in patterns)
poispatterns = [sample_pp(PoissonProcess(L),W) for _ in 1:noriginals]
#poispatterns = [sample_pp(PoissonProcess(L),W) for _ in 1:200]

#########################################
# Analyze Ni distributions
#########################################

Nvectors = Array{Vector{Int64}}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    s = patterns[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    Nvectors[i,j] = [N(s,box) for box in boxes]
end

poisNvectors = Array{Vector{Int64}}(undef, (length(poispatterns),nresolutions))
for i in eachindex(poispatterns), j in 1:nresolutions
    s = poispatterns[i]
    boxes = PointProcessLearning.partitions(W,resolutions[j])
    poisNvectors[i,j] = [N(s,box) for box in boxes]
end

using Distributions
function plot_Nvectors(j::Int64,Nvectors::Array{Vector{Int64}})
    res = resolutions[j]
    plt = plot(;title="Distr. of N_i (n=$res)",
            xlabel="Number of points", ylabel="occurrences")
    #for i in 1:noriginals
    #    hist = countmap(Nvectors[i,j])
    #    scatter!(plt, collect(keys(hist)).-0.15.+0.3*rand(), collect(values(hist)).-0.15.+0.3*rand(), labels=false, mc=cscheme[2], opacity=0.5)
    #end

    hist_Nvector = countmap(vcat([Nvectors[i,j] for i in 1:noriginals]...))
    mean_hist = Dict(k=>v/noriginals for (k,v) in hist_Nvector)
    plot!(plt, mean_hist, labels="mean count", linewidth = 4, color=cscheme[2])
    bar!(plt, mean_hist, labels="mean count", linewidth = 0, color=cscheme[2], opacity=0.5)
    Nmean = mean([Nvectors[i,1][1] for i in 1:noriginals])
    L = Nmean/((resolutions[j])^2)
    nvals = collect(0:maximum(collect(keys(mean_hist)))) 
    #plot!(plt,nvals, (resolutions[j])^2*pdf.(Poisson(L),nvals), linewidth=3, linestyle=:dash, color=:red, label="Poisson")
    plot!(plt,nvals, (resolutions[j])^2*pdf.(Poisson(L),nvals), linewidth=3, linestyle=:dash, color=:red, label="Poisson")
    return plt
end

plotlist = [plot_Nvectors(j,Nvectors) for j in 1:nresolutions]
plot(reverse(plotlist)..., layout=(nresolutions,1))
plot!(size=(2700/7,2000))
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_Ndistribution.pdf")





### Distribution of distances for Nvectors
countdistances = Array{Float64}(undef, (nresolutions, Int(noriginals*(noriginals-1)/2)))
for k in 1:nresolutions
    dists = []
    for i in 1:noriginals-1
        for j in i+1:noriginals
            distij = sqrt(sum((Nvectors[i,k] .- Nvectors[j,k]).^2))
            push!(dists,distij)
        end
    end
    countdistances[k,:] = dists
end
plts = [histogram(countdistances[k,:]) for k in 1:nresolutions]
plot(plts...)
qtls = [quantile(countdistances[k,:], 0.95) for k in 1:nresolutions]





################################
## Correlation analysis
################################

# Adjacency matrix of n by m grid 
function adjacencymat(n::Int,m::Int)
    # dummy matrix....ugly hack
    B = Matrix{Int64}(undef,(n,m))
    x = reshape(CartesianIndices(B),n*m)
    A = Matrix{Int64}(undef,(length(x),length(x)))
    for i in eachindex(x), j in eachindex(x)
        dif = x[i]-x[j]
        A[i,j] = abs(dif[1])+abs(dif[2]) == 1 ? 1 : 0
    end
    return A
end


function MoranIndex(x; removebias=false)
    n = length(x)
    A = adjacencymat(Int(sqrt(n)),Int(sqrt(n)))
    xbar = mean(x)
    dx = x .- xbar
    val = dx'*A*dx
    #val = sum( A[i,j]*(x[i]-xbar)*(x[j]-xbar)  for i in eachindex(x), j in eachindex(x))
    val = val/sum(A)
    val = val*n/sum(dx.^2)
    if removebias==true
        return val - (-1/(n-1))
    else
        return val
    end
end

MoranBias(n) = -1/(n-1)


MIvals = [mean(MoranIndex.(Nvectors[:,j])) for j in 1:nresolutions]
MIvals = MIvals .- MoranBias.(resolutions.^2)
#plot(resolutions,MIvals)
plot([1/res for res in resolutions],MIvals,
    labels="M(r)",
    xlabel="side length r",
    ylabel="Moran Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
#plot!([1/res for res in resolutions],[MoranBias(res^2) for res in resolutions])
#vline!([R,2*R], linestyle=:dash, labels="R,2R")

poisMIvals = [mean(MoranIndex.(poisNvectors[:,j])) for j in 1:nresolutions]
poisMIvals = poisMIvals .- MoranBias.(resolutions.^2)
#plot(resolutions,poisMIvals)
plot!([1/res for res in resolutions],poisMIvals,
    labels="M(r) (Poisson)",
    xlabel="side length r",
    ylabel="Moran Index",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2R")
hline!([0], linestyle=:dot, labels=false)
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_MoranIndex.pdf")


################################
## Chisq
################################

####### chisquared Test
X² = Array{Float64}(undef, (noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Nvector = Nvectors[i,j]
    Ntot = sum(Nvector)
    Ei = Ntot/measure(W)
    Oi = Nvector
    X²[i,j] = sum((Oi .- Ei).^2 ./ Ei)
end
meanX² = vec(mean(X²,dims=1))
plot([1/n for n in resolutions[2:nresolutions]],
    [1-cdf(Chisq(resolutions[j]^2-1),meanX²[j]) for j in 2:nresolutions], labels="p: 1-CDF of Chisq(n²-1) at X²",
    xlabel="side length r",
    ylabel="p value", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_chisqtest_pvals.pdf")

plot([1/n for n in resolutions[2:nresolutions]],[meanX²[j]/quantile(Chisq(resolutions[j]^2-1),0.99) for j in 2:nresolutions],
    labels="X²/χ²(n²-1,0.99)",
    xlabel="side length r",
    ylabel="", color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")

########################
## Dispersion Index
########################

Dmat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions 
    Nvector = Nvectors[i,j]
    Dmat[i,j] = var(Nvector)/mean(Nvector)
end
D = vec(mean(Dmat, dims=1))

#D = [mean(var2.(Nvectors[:,j])./mean.(Nvectors[:,j])) for j in 1:nresolutions]
plot(resolutions,D)
plot([1/n for n in resolutions],D,
    labels="D(r)",
    xlabel="side length r",
    ylabel="Dispersion Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")
#savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_disperson_index_raw.pdf")

poisDmat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions 
    Nvector = poisNvectors[i,j]
    poisDmat[i,j] = var(Nvector)/mean(Nvector)
end
poisD = vec(mean(poisDmat, dims=1))

plot!([1/n for n in resolutions],poisD,
    labels="D(r) pois",
    xlabel="side length r",
    ylabel="Dispersion Index",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2*R")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_disperson_index_raw.pdf")


########################
## Morisita Index
########################

function MorisitaIndexMat(Nvectors)
    Mmat = Matrix{Float64}(undef,(noriginals,nresolutions)) 
    for i in 1:noriginals, j in 1:nresolutions
        Nvector = Nvectors[i,j]
        Ntot = sum(Nvector)
        k = resolutions[j]^2
        Mmat[i,j] = k*sum(Nvector[i]*(Nvector[i]-1) for i in eachindex(Nvector))/(Ntot*(Ntot-1))
    end
    return Mmat 
end

Mmat = MorisitaIndexMat(Nvectors)
mean_Mindex = vec(mean(Mmat,dims=1)) 
plot([1/n for n in resolutions], mean_Mindex, 
    labels="M(r)",
    xlabel="side length r",
    ylabel="Morisita Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2R")

# same for Poisson
Mmat = MorisitaIndexMat(poisNvectors)
mean_Mindex = vec(mean(Mmat,dims=1)) 
plot!([1/n for n in resolutions], mean_Mindex, 
    labels="M(r) (poisson)",
    xlabel="side length r",
    ylabel="Morisita Index",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], linestyle=:dash, labels="R,2R")
savefig("~/Dropbox/Statistics Education/Examensarbete/Notes/GeyerProcess_morisita_index_raw.pdf")


########################
## GreigSmith Index
########################


function partitions(W::Box,dim::Tuple{Int64,Int64})
    lo, up = coordinates.(extrema(W))
    dx = (up[1]-lo[1])/dim[1]
    dy = (up[2]-lo[2])/dim[2]
    boxmat = [Box((lo[1]+k*dx,lo[2]+j*dy),(lo[1]+(k+1)*dx,lo[2]+(j+1)*dy)) for k=0:(dim[1]-1), j=0:(dim[2]-1)]
    return reshape(boxmat,dim[1]*dim[2])
end

function GreigSmith(s::PointSet,W::Box,nmax::Int)
    q = Int(log2(nmax^2))
    svec = Array{Float64}(undef,q-1)
    Nvec = Array{Float64}(undef,q-1)
    n = nmax
    m = nmax
    #testvec = Array{Tuple{Int,Int}}(undef,q-1)
    for i in 1:q-1
        # inner boxes
        Ninner = [N(s,box) for box in partitions(W,(n,m))]
        Nvec[i] = mean(Ninner)
        # outer boxes
        n = (i%2==0) ? Int(n/2) : n
        m = (i%2==0) ? n : Int(n/2)
        Nouter = [N(s,box) for box in partitions(W,(n,m))]
        svec[i] = sum(Ninner.^2) - 0.5*sum(Nouter.^2)
    end
    Ivec = svec./Nvec
    Xvec = Ivec./[2^(q-i) for i in 1:q-1]
    #return svec, Ivec, Xvec
    return Ivec
end

q = Int(log2(64^2))
GSImat = Array{Float64}(undef,(noriginals,q-1))
for i in 1:noriginals
    GSImat[i,:] = GreigSmith(patterns[i],W,64)
end
GSIvals = vec(mean(GSImat,dims=1))

plot(GSIvals)
plot(GSIvals./[quantile(Chisq(2^(q-i)),0.5) for i in 1:q-1])
plot([1/(2^(q-i)) for i in 1:q-1],GSIvals./[quantile(Chisq(2^(q-i)),0.5) for i in 1:q-1])

X2vals = GSIvals./[2^(q-i) for i in 1:q-1]
plot([1/(2^(q-i)) for i in 1:q-1], X2vals)
#plot([(i+1)/64 for i in 1:q-1], Xvec)
vline!([R,2*R])


########################
## Entropy Index
########################

function EntropyOneStep(s::PointSet, b::Box, n::Int)
    largerboxes = PointProcessLearning.partitions(b,n)
    entropyvector = Array{Float64}(undef,n^2)
    for i in 1:n^2
        lbox = largerboxes[i]
        Ntot = N(s,lbox)
        if Ntot>0
            Nvals = [N(s,sbox) for sbox in PointProcessLearning.partitions(lbox,2)]
            pvals = Nvals/Ntot
            entropyvector[i] = -sum(p>0 ? p*log(p) : 0 for p in pvals)
        else 
            entropyvector[i] = 0
        end
    end
    return mean(entropyvector)
end


Emat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Emat[i,j] = EntropyOneStep(patterns[i],W,resolutions[j])
end
Evals = vec(mean(Emat, dims=1))
plot([1/(2*n) for n in resolutions],Evals .- maxentropy(2), 
    labels="Entropy",
    xlabel="side length r",
    ylabel="Entropy",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)

# same for poisson
Emat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Emat[i,j] = EntropyOneStep(poispatterns[i],W,resolutions[j])
end
Evals = vec(mean(Emat, dims=1))
plot!([1/(2*n) for n in resolutions[1:nresolutions]],Evals.- maxentropy(2), 
    labels="Entropy (pois)",
    xlabel="side length r",
    ylabel="Entropy",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)


function Entropy(s::PointSet,b::Box,n::Int64)
    Ntot = N(s,b)
    entropy = 0 
    if Ntot>0
        Nvals = [N(s,box) for box in PointProcessLearning.partitions(b,n)]
        pvals = Nvals/Ntot
        entropy = -sum(p>0 ? p*log(p) : 0 for p in pvals)
    end
    return entropy
end

maxentropy(n::Int) = -n^2*(1/n^2)*log(1/n^2)

Emat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Emat[i,j] = Entropy(patterns[i],W,resolutions[j])
end
Evals = vec(mean(Emat, dims=1))
plt1 = plot([1/(2*n) for n in resolutions[1:nresolutions]],Evals, 
    labels="Entropy",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)
plot!(plt1, [1/(2*n) for n in resolutions],maxentropy.(resolutions),
    linestyle=:dash, label="th. independent")

plt3 = plot([1/(2*n) for n in resolutions],Evals .- maxentropy.(resolutions), 
    labels="Entropy",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)


plt2 = plot([1/(2*n) for n in resolutions[1:nresolutions]],[Evals[i]/log(resolutions[i]^2) for i in 1:nresolutions], 
    labels="Entropy Index",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)

# Same for pois
Emat = Array{Float64}(undef,(noriginals,nresolutions))
for i in 1:noriginals, j in 1:nresolutions
    Emat[i,j] = Entropy(poispatterns[i],W,resolutions[j])
end
Evals = vec(mean(Emat, dims=1))
plot!(plt1,[1/(2*n) for n in resolutions[1:nresolutions]],Evals, 
    labels="Entropy (pois)",
    xlabel="side length r",
    ylabel="Entropy",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)

plot!(plt2,[1/(2*n) for n in resolutions[1:nresolutions]],[Evals[i]/log(resolutions[i]^2) for i in 1:nresolutions], 
    labels="Entropy Index (pois)",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[1],
    marker = (3,cscheme[1],5,stroke(1,:black)))
vline!([R,2*R], labels="R,2R", color=:red, linestyle=:dash)

plot!(plt3,[1/(2*n) for n in resolutions],Evals .- maxentropy.(resolutions), 
    labels="Entropy",
    xlabel="side length r",
    ylabel="Entropy Index",
    color=cscheme[2],
    marker = (3,cscheme[2],5,stroke(1,:black)))


plot(plt1,plt2)

plt1
plot!(plt1,[1/(2*n) for n in resolutions],maxentropy.(resolutions))


maxentropy.(resolutions)
maxentropy.(resolutions)./log.(resolutions.^2)






## Variogram

#########################################
# Set up Area interaction process
#########################################
# Parameters
#R = 0.025;
R = 0.075
η = 40 # η = γ^(π*R^2)
#λ₀ = 4000; 
#β = λ₀/η
β = 250
# Interaction function
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
# Set up exponential model (parameters and covariates)
θ₀ = [log(β),log(η)]
S = [(u,s)->1, (u,s)->-C(u,s)]
# Papangelou conditional intensity
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
# Define point Gibbs process and sample one pattern
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=100_000)
plot(s,W)


function distmat(boxes)
    centers = centroid.(boxes)
    dists = Matrix{Float64}(undef,(length(centers),length(centers)))
    for i in eachindex(centers), j in eachindex(centers)
        dists[i,j] = dist(centers[i],centers[j])
    end
    return dists
end

jval = 6
#Nvector = Nvectors[2,jval]
boxes = PointProcessLearning.partitions(W,resolutions[jval])
distm = distmat(boxes)
rvals = sort(unique(distm))
vmeans = zeros(length(rvals),length(Nvectors[:,jval]))
vvars = zeros(length(rvals),length(Nvectors[:,jval]))
for i in eachindex(Nvectors[:,jval])
    for (k,r) in enumerate(rvals)
        idx = findall(isequal(r),distm)
        Nvector = Nvectors[i,jval]
        vmeans[k,i] = mean(0.5*(Nvector[ind[1]]-Nvector[ind[2]])^2 for ind in idx)
        vvars[k,i] = var(0.5*(Nvector[ind[1]]-Nvector[ind[2]])^2 for ind in idx)
    end
end
# run from here
mean(vmeans, dims=2)
plot(rvals,vec(mean(vmeans, dims=2)), yerr=vec(mean(vvars,dims=2)))
plot(rvals[2:end],vec(mean(vmeans, dims=2))[2:end])
vline!([R])
plot!(rvals, x->(0.0305*(1-exp(-x^2/(0.3333*R)^2))))

vmeans2 = vec(mean(vmeans,dims=2))
vvars2 = vec(mean(vvars,dims=2))
using LsqFit

γ(r,σ₀,σ,θ) = σ.*(1 .- exp.(-r./θ)) .+ σ₀
γ(r,T) = γ(r,T[1],T[2],T[3])

γ([0.25,0.5],T0)

rvals
vmeans2
wt = 1 ./ vmeans2
T0 = [0.1,0.5,R]
fit = curve_fit(γ,rvals[2:50],vmeans2[2:50],wt[2:50],T0)
fit = curve_fit(γ,rvals[2:50],vmeans2[2:50],T0)
fit = curve_fit(γ,rvals[2:end],vmeans2[2:end],T0)
fit.param
plot!(rvals[2:50],γ(rvals[2:50],fit.param))

S
res = EstimateParamsPL_Logit(S,s,W)
exp.(coef(res))










#########################################
# Set up Geyer process
#########################################
# Parameters
# Distance between points
R = 0.1; 
Nmin = 40; perc = 100_000; Nmax = Nmin*perc;
λ₀=Nmin/measure(W); γ₀ = 10;
sat=log(perc)/log(γ₀)
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
#λ₀ = 100; g₀ = 1.25;
θ₀ = [log(λ₀),log(g₀)]
S = [(u,s)->1, (u,s)->min(sat,t(u,s))]
#deleteat!(S,1)
# For fixed finite process
#θ₀ = [log(g₀)]
#S = [(u,s)->min(sat,t(u,s))]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
#p = FiniteGibbsProcess2(logλ,200)
s = sample_pp(p,Wplus)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=200_000)
#s = sample_pp2(Random.GLOBAL_RNG,p,Wplus; niter=50_000)
plot(s,W)
CondIntPlot(logλ,s,W)