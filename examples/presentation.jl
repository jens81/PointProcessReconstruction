## Remember to run Pkg> activate .
using Meshes, Plots, StatsBase, GLM, Combinatorics
import Random
include("../src/PointProcessLearning.jl")
using .PointProcessLearning
import ColorSchemes
import ColorSchemes.okabe_ito
cscheme = okabe_ito

#########################################
# Set up window
#########################################
W = Box((0.,0.),(1.,1.))
Wplus = PointProcessLearning.enlarge(W,0.1)

# Resolutions
resolutions = [1, 2, 4, 8, 16, 32,64]
boxes = [PointProcessLearning.partitions(W,res) for res in resolutions]




# Covariate density (random)
ngauss = 3
μ = [rand(2) for _ in 1:ngauss]
σx = [sqrt(0.05*rand()) for _ in 1:ngauss]; 
σy = [sqrt(0.05*rand()) for _ in 1:ngauss];
w = [0.5+rand() for _ in 1:ngauss];
fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy^2))
f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:ngauss)
#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour(range(0,1,length=20),range(0,1,length=20),f)
M = sum(w)
M = 1.5


f(x,y) = x
contour(range(0,1,length=20),range(0,1,length=20),f)

#########################################
# Set up Inhomogeneous Poisson process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 120; g₀ = 1;# 1.25;
δ = 5;
S₂(u) = f(coordinates(u)[1],coordinates(u)[2]);
#θ₀ = [log(λ₀),log(δ),log(g₀)]
#S = [(u,s)->1,(u,s)->S₂(u),(u,s)->min(sat,t(u,s))]
θ₀ = [log(λ₀),log(δ)]
S = [(u,s)->1,(u,s)->S₂(u)]

# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=100_000)
contourf(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1, aspect_ratio=1)
plot!(s,W, aspect_ratio=1, alpha=0, title="covariate density, S₁(u)")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/inhom_cond_int_covariate.png")
plt = CondIntPlot(logλ,s,W,N=100)
title!(plt, "sample and intensity, λ(u)")
savefig(plt,"/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/inhom_cond_int_sample.png")
plt = IntPlot(logλ,s,W,N=100)
title!(plt,"intensity, λ(u)")
savefig(plt,"/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/inhom_cond_int.png")


originals = [sample_pp(Random.GLOBAL_RNG,p,W,niter=50_000) for _ in 1:200]

function IntPlot(logλ::Function,s,W::Box; N=50)
    λ(u,s) = exp(logλ(u,s))
    lo, up = coordinates.(extrema(W))
    u1 = range(lo[1],stop=up[1],length=N)
    u2 = range(lo[2],stop=up[2],length=N)
    z = [λ(Point([x,y]),s) for x in u1, y in u2]
    plt = heatmap(u1,u2,z', color=cgrad([:white,:red]))
    plot!(plt,s,W, mc=:black, alpha=0)
    return plt
end


#IntPlot(logλ,s,W;N=16)

k = 2
res = resolutions[k]
a = round(1/res,digits=3)
A = measure(boxes[k][1])
Nvector = [mean(N(orig,b) for orig in originals) for b in boxes[k]]
lo, up = coordinates.(extrema(W))
u1 = unique(map(b->coordinates(centroid(b))[1],boxes[k]))
u2 = unique(map(b->coordinates(centroid(b))[2],boxes[k]))
z = reshape(Nvector./A, (Int(sqrt(length(Nvector))),Int(sqrt(length(Nvector)))))
heatmap(u1,u2,z', color=cgrad([:white,:red]))
plot!(PointProcessLearning.Box2Shape.(boxes[k]), linestyle=:dash, fillcolor=false)
plot!(s,W, mc=:black, alpha=0, title="k=$res, a=$a")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/inhom_cond_int_rec_k2.pdf")
savefig("/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/inhom_cond_int_rec_k2.png")




#########################################
# Set up Inhomogeneous Poisson process
#########################################
# Parameters
# Distance between points
R = 0.05; sat=4;
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
# Parameters
λ₀ = 120; g₀ = 1.25;# 1.25;
δ = 1;
S₂(u) = f(coordinates(u)[1],coordinates(u)[2]);
θ₀ = [log(λ₀),log(δ),log(g₀)]
S = [(u,s)->1,(u,s)->S₂(u),(u,s)->min(sat,t(u,s))]
#θ₀ = [log(λ₀),log(δ)]
#S = [(u,s)->1,(u,s)->S₂(u)]
# Generate patter
logλ(u,s,θ) = sum(θ[i]S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=100_000)
plt = CondIntPlot(logλ,s,W,N=200)
title!(plt,"Geyer, λ(u;x)")
savefig(plt,"/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/geyer_condint.pdf")


#########################################
# Set up Area interaction process
#########################################
# Parameters
#R = 0.025;
R = 0.05
η = 2.7 # η = γ^(π*R^2)
λ₀ = 500; β = λ₀/η
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
s = sample_pp(Random.GLOBAL_RNG,p,Wplus,niter=60_000)
plt = CondIntPlot(logλ,s,W;N=150)
title!(plt,"Area interaction, λ(u;x)")
savefig(plt,"/Users/jensmichelsen/Dropbox/Statistics Education/Examensarbete/presentation/chalmers-master/figures/area_condint.pdf")
