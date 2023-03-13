## Remember to run Pkg> activate .
#using Meshes
using Plots
import Random
#using StatsBase
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

3+4

# Set up window
W = Box((0.,0.),(1.,1.))

# Binomial Point Process
p1 = BinomialProcess(50)
s1 = sample_pp(p1,W)
plot(s1,W)

# Poisson Point Process
λ₂ = 100
p2 = PoissonProcess(λ₂)
s2 = sample_pp(p2,W)
plot(s2,W)

# Inhomogeneous Point Process
λ₃(x::Point) = exp(5+2*coordinates(x)[1])
p3 = PoissonProcess(λ₃)
s3 = sample_pp(p3,W)
plot(s3,W)

# Gibbs Point Process (Strauss)
λ₀ = 200
γ = 0.8
R = 0.1
logλ4(u::Point,s::PointSet) = log(λ₀) + log(γ)*(NNdist(s ∪ u,R) - NNdist(s,R))
p4 = GibbsProcess(logλ4)
s4 = sample_pp(p4,W)
plot(s4,W)
CondIntPlot(logλ4,s4,W)

# Gibbs Point Process (Geyer)
λ₀ = 50
γ = 1.3
R = 0.2
sat = 8
# Distance between points
#dist(x::Point,y::Point) = sqrt((coordinates(x)[1]-coordinates(y)[1])^2+(coordinates(x)[2]-coordinates(y)[2])^2)
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ5(u::Point,s::PointSet) = log(λ₀) + log(γ)*min(sat,t(u,s))
p5 = GibbsProcess(logλ5)
s5 = sample_pp(p5,W)
plot(s5,W)
StoyanGrabarnik(s5,logλ5,W)
StoyanGrabarnikPlot(s5,logλ5,W,4)


# Gibbs Point Process (Inhomogeneous Geyer)
λ₆(u) = exp(4+2.5*coordinates(u)[1])
γ = 1.3
R = 0.05
sat = 10
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ6(u::Point,s::PointSet) = log(λ₆(u)) + log(γ)*min(sat,t(u,s))
p6 = GibbsProcess(logλ6)
s6 = sample_pp(p6,W)
plot(s6,W,axis=false,ticks=false)


# Area Interaction Process
λ₇ = 150
γ = 1.2
R = 0.05
U(s) = BallUnionArea(s,R,W;N=15) # just won't fucking work!!!
#U(s) = BallUnionArea2(s,R,W)
logλ7(u::Point,s::PointSet) = log(λ₇) - log(γ)*(U(s ∪ u)-U(s))
p7 = GibbsProcess(logλ7)
s7 = sample_pp(Random.GLOBAL_RNG,p7,W; niter=1_000, progress=true)
plot(s7,W,axis=false,ticks=false)
CondIntPlot(logλ7,s7,W)


