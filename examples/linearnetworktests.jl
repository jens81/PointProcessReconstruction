using Meshes
using Plots
include("../src/PointProcessLearning.jl")
using .PointProcessLearning

W = Box((0.,0.),(1.,1.))
LR = RandomLines(30,PointProcessLearning.reduce(W,0.05))
L = LinearNetwork(LR)
PlotLinearNetwork(L)
p = BinomialProcess(50)
s,segids = sample_pp(p,L)
plot!(s,W)

tL = thin(L,0.75)
PlotLinearNetwork(tL)
p = BinomialProcess(100)
s = sample_pp(p,tL)
plot!(s,W)


function enlarge(b::Box, r::Real)
    lo, up = coordinates.(extrema(b))
    lo = Tuple(lo .- r)
    up = Tuple(up .+ r)
    return Box(lo,up)
end

function reduce(b::Box, r::Real)
    lo, up = coordinates.(extrema(b))
    lo = Tuple(lo .+ r)
    up = Tuple(up .- r)
    return Box(lo,up)
end

W = enlarge(W,0.001)


function subnetwork(L::LinearNetwork,b::Box)
    r0 = 0.001
    b = enlarge(b,r0)
    boxvertices = vertices(b)
    boxsegments = [Segment(boxvertices[i],boxvertices[i+1]) for i in 1:3]
    push!(boxsegments,Segment(boxvertices[1],boxvertices[4]))
    V = Point2[]
    E = Tuple{Int64,Int64}[]
    for i in 1:length(L.segments)
        s = L.segments[i]
        v = vertices(s)
        vinb = filter(v->in(v,b) ,vertices(s))
        if length(vinb) == 2
            push!(V,vertices(s)...)
            push!(E,(length(V)-1,length(V)))
        elseif length(vinb) == 1
            vinb = vinb[1]
            intpoints = [s ∩ sb for sb in boxsegments]
            intpoints = filter(p->!isnothing(p),intpoints)
            intp = intpoints[1]
            push!(V,vinb,intp)
            push!(E,(length(V)-1,length(V)))
        else
            intpoints = [s ∩ sb for sb in boxsegments]
            intpoints = filter(p->!isnothing(p),intpoints)
            if length(intpoints) == 2
                push!(V,intpoints...)
                push!(E,(length(V)-1,length(V)))
            end
        end
        #intpoints = [s ∩ sb for sb in boxsegments]
        #intpoints = filter(p->!isnothing(p),intpoints)
        #println(intpoints)
        #if length(intpoints) == 2
        #    push!(V,intpoints...)
        #    push!(E,(length(V)-1,length(V)))
        #elseif length(intpoints) == 1
        #    intp = intpoints[1]
        #    v = filter(v->in(v,b) ,vertices(s))[1]
        #    push!(V,v,intp)
        #    push!(E,(length(V)-1,length(V)))
        #end
    end
    return LinearNetwork(V,E)
end


p1 = PlotLinearNetwork(L)
Wsub1 = Box((0.0,0.0),(0.5,0.5))
Wsub2 = Box((0.5,0.0),(1.0,0.5))
Wsub3 = Box((0.0,0.5),(0.5,1.0))
Wsub4 = Box((0.5,0.5),(1.0,1.0))
#Wsub1 = Box((0.0-r0,0.0-r0),(0.5+r0,0.5+r0))
#Wsub2 = Box((0.5-r0,0.0-r0),(1.0+r0,0.5+r0))
#Wsub3 = Box((0.0-r0,0.5-r0),(0.5+r0,1.0+r0))
#Wsub4 = Box((0.5-r0,0.5-r0),(1.0+r0,1.0+r0))
Lsub1 = subnetwork(L,Wsub1)
Lsub2 = subnetwork(L,Wsub2)
Lsub3 = subnetwork(L,Wsub3)
Lsub4 = subnetwork(L,Wsub4)
PlotLinearNetworkW!(p1,Lsub1,W)
PlotLinearNetworkW!(p1,Lsub2,W)
PlotLinearNetworkW!(p1,Lsub3,W)
PlotLinearNetworkW!(p1,Lsub4,W)

s = sample_pp(p,Lsub2)
plot!(s,W)

function PlotLinearNetworkW!(plt,L::LinearNetwork,W::Box; show_vertices=false)
    lo, up = coordinates.(extrema(W))
    if show_vertices
        pts = PointSet(L.vertices)
        W = boundingbox(pts)
        plot!(plt,pts,W)
    end
    for s in L.segments
        v = vertices(s)
        v1 = coordinates(v[1])
        v2 = coordinates(v[2])
        x = [v1[1],v2[1]]
        y = [v1[2],v2[2]]
        plot!(plt,x,y, label=false, line=(0.5,4,:red), xlim=(lo[1],up[1]), ylim=(lo[2],up[2]))
    end
    return plt
end


# Gibbs Point Process (Geyer)
λ₀ = 10
γ = 1.2
R = 0.05
sat = 3
# Distance between points
#dist(x::Point,y::Point) = sqrt((coordinates(x)[1]-coordinates(y)[1])^2+(coordinates(x)[2]-coordinates(y)[2])^2)
t(u::Point,s::PointSet) = sum(dist(u,x)<R for x in s.items)
logλ5(u::Point,s::PointSet) = log(λ₀) + log(γ)*min(sat,t(u,s))
p5 = GibbsProcess(logλ5)
s5 = sample_pp(p5,L)
PlotLinearNetwork(L)
plot!(s5,W)

# Covariate density
μ = [rand(2) for _ in 1:4]
σx = [sqrt(0.05*rand()) for _ in 1:4]; 
σy = [sqrt(0.05*rand()) for _ in 1:4];
w = [0.5+rand() for _ in 1:4];

fN(x,y,μ,σx,σy,w) = w*exp(-(x-μ[1])^2/(2*σx^2) - (y-μ[2])^2/(2*σy))

f(x,y) = sum(fN(x,y,μ[i],σx[i],σy[i],w[i]) for i in 1:3)

#plot(range(0,1,length=20),range(0,1,length=20),f,st=:surf)
contour(range(0,1,length=20),range(0,1,length=20),f)

#### Cheat
#L = LinearNetwork(Point2[],Tuple{Int64,Int64}[],Matrix{Int64}(undef,(1,1)),LR)


# Try to create point pattern
#
R = 0.025;
#γ = 10e18; η = γ^(π*R^2)
η = 1.25
λ₀ = 5; β = λ₀/η
δ = 3
S3(u::Point) = f(coordinates(u)[1],coordinates(u)[2])
#
θ₀ = [log(β),log(η),log(δ)]
C(u::Point,s::PointSet) = FractionOfContestedArea(u,s,R; nd=25)
S = [(u,s)->1, (u,s)->-C(u,s),(u,s)->S3(u)]
# Generate patter
logλ(u,s,θ) = sum(θ[i]*S[i](u,s) for i in eachindex(θ))
logλ(u,s) = logλ(u,s,θ₀)
p = GibbsProcess(logλ)
#s = sample_pp(Random.GLOBAL_RNG,p,Wplus; niter=20_000)
s,segids = sample_pp(p,L; niter=50_000)
PlotLinearNetwork(L)
contourf!(range(0,1,length=20), range(0,1,length=20),f, opacity=0.1)
plot!(s,W)

using GLM
res = EstimateParamsPL_Logit(S,s,L; nd=10_000)
exp.(coef(res))

StoyanGrabarnik(s,logλ,L,segids)