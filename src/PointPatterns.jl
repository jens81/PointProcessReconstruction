
"""
    PointProcess
A spatial point process.
"""
abstract type PointProcess end

################# Helper functions

"""
    N(s,b)
A function to count the number of points of set 's' inside box 'b'
"""
N(s::PointSet,b::Box) = sum(x∈b for x in s.items)

# Distance between points
dist(x::Point,y::Point) = sqrt((coordinates(x)[1]-coordinates(y)[1])^2+(coordinates(x)[2]-coordinates(y)[2])^2)

# Define unions
Base.union(s1::PointSet,s2::PointSet) = PointSet(vcat(s1.items,s2.items)...)
Base.union(s::PointSet,u::Point) = PointSet(vcat(s.items,u)...)
Base.intersect(s1::PointSet,s2::PointSet) = PointSet(s1.items ∩ s2.items)
\(s1::PointSet,s2::PointSet) = PointSet(setdiff(s1.items,s2.items))

"""
    NNdist(s,r)
A function to count the number of pairs of points in 's' within a distance of 'r'.
"""
function NNdist(X::Vector{Point2}, r::Real) 
    n = 0
    N(X) = length(X)
    for i in 1:N(X)
        for j in (i+1):N(X)
            if dist(X[i],X[j])<r
                n += 1
            end
        end
    end
    return n
end 

NNdist(s::PointSet, r::Real) = NNdist(s.items, r)


# Running models
struct model_pp{T<:Vector,Lo<:Vector,Up<:Vector,logL<:Function,ind<:Bool}
    θ :: T
    lo :: Lo
    up :: Up
    logλ :: logL
    independent :: ind
end


################# Point Process definitions and sampling

#--------------------
# Binomial Point Process
#--------------------

"""
    BinomialProcess(n)
A Binomial point process with `n` points.
"""
struct BinomialProcess <: PointProcess
  n::Int
end

function sample_pp(rng::Random.AbstractRNG,
    p::BinomialProcess, b::Box{Dim,T}) where {Dim,T}
    # region configuration
    lo, up = coordinates.(extrema(b))

    # product of uniform distributions
    U = product_distribution([Uniform(lo[i], up[i]) for i in 1:Dim])

    # return point pattern
    return PointSet(rand(rng, U, p.n))
end


function BinomialRejectionSampling(n::Integer, f::Function, b::Box,M::Real)
    # region configuration
    lo, up = coordinates.(extrema(b))
    V = measure(b)
    # product of uniform distributions
    U = product_distribution([Uniform(lo[i], up[i]) for i in 1:2])
    s = EmptyPointSet()
    while length(s.items) < n 
        p = rand()
        u = Point(rand(U))
        while p > f(u)/M #f(u)/(M/V)
            p = rand()
            u = Point(rand(U))
        end
            s = s ∪ u
    end
    return s
end



#--------------------
# Poisson Point Process
#--------------------

"""
   PoissonProcess(λ)
A Poisson process with intensity `λ`.
"""
struct PoissonProcess{L<:Union{Real,Function}} <: PointProcess
  λ::L
end

Base.union(p₁::PoissonProcess{<:Real}, p₂::PoissonProcess{<:Real}) =
  PoissonProcess(p₁.λ + p₂.λ)

Base.union(p₁::PoissonProcess{<:Function}, p₂::PoissonProcess{<:Function}) =
  PoissonProcess(x -> p₁.λ(x) + p₂.λ(x))

Base.union(p₁::PoissonProcess{<:Real}, p₂::PoissonProcess{<:Function}) =
  PoissonProcess(x -> p₁.λ + p₂.λ(x))

Base.union(p₁::PoissonProcess{<:Function}, p₂::PoissonProcess{<:Real}) =
  PoissonProcess(x -> p₁.λ(x) + p₂.λ)

function sample_pp(rng::Random.AbstractRNG,
    p::PoissonProcess{<:Real}, b::Box)
    # region configuration
    lo, up = coordinates.(extrema(b))

    # simulate number of points
    λ = p.λ; V = measure(b)
    n = rand(rng, Poisson(λ*V))

    # product of uniform distributions
    U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])

    # return point pattern
    return PointSet(rand(rng, U, n))
end

function sample_pp(rng::Random.AbstractRNG,
    p::PoissonProcess{<:Real}, L::LinearNetwork)

    # simulate number of points
    λ = p.λ; V = measure(L)
    n = rand(rng, Poisson(λ*V))

    # product of uniform distributions
    pbin = BinomialProcess(n)
    s = sample_pp(pbin,L)
    # return point pattern
    return s
end

#--------------------
# Poisson Point Process : INHOMOGENEOUS CASE
#--------------------

function sample_pp(rng::Random.AbstractRNG,
    p::PoissonProcess{<:Function}, b::Box; Λmax = 10000)
    
    # Generate homogeneous sample
    p₀ = PoissonProcess(Λmax)
    pp = sample_pp(rng, p₀, b)
    
    # Thinning
    λ = p.λ
    pr(x) = λ(x)/Λmax
    X = pp.items
    X = X[ [ rand()<pr(x)  for x in X] ]

    # return point pattern
    return PointSet(X...)
end


#--------------------
# Gibbs Point Process
#--------------------

"""
   GibbsProcess(logλ)
A Gibbs Process with Papangelou conditional log intensity 'logλ'.
"""
struct GibbsProcess{logL<:Function} <: PointProcess
    logλ::logL
end


# Birth death move
function sample_pp(rng::Random.AbstractRNG,
    p::GibbsProcess{<:Function}, 
    b::Box; niter=10_000, progress=false)

    # region configuration
    lo, up = coordinates.(extrema(b))
    V = measure(b)

    # Number of points
    N(S) = length(S.items)

    # product of uniform distributions
    U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])
    I(n) = DiscreteUniform(1,n) 

    # probabilities
    logλ(u,S) = N(S)>0 ? p.logλ(u,S) : 0
    r(u,S) = exp(logλ(u,S)+log(V)-log(N(S)+1))
    pbirth = 0.5
    pmove = 0.5
    rmove(u,ξ,S₋) = pmove*exp(logλ(u,S₋)-logλ(ξ,S₋))/(N(S₋)+1)

    # Initial sample
    S = sample_pp(BinomialProcess(100),b)
    #println("Performing ",niter," steps")
    for m in 1:niter
        if progress
            if mod(m,100)==0
                println("step ",m)
            end
        end
        Sprop = deepcopy(S)
        if rand() < pmove
            # Randomly select and move point
            i = rand(I(N(S)))
            ξ = Sprop.items[i]
            deleteat!(Sprop.items,i)
            u = Point(rand(U))
            # accept/reject move
            S = rand()<rmove(u,ξ,Sprop) ? Sprop ∪ u : S  # EDIT THIS!
        elseif rand() < pbirth
            # Randomly generate new point
            u = Point(rand(U))
            Sprop = Sprop ∪ u
            # accept/reject birth
            S = rand()<r(u,S) ? Sprop : S 
        elseif N(S)>1
            # randomly select and remove point
            i = rand(I(N(S)))
            deleteat!(Sprop.items,i)
            u = S.items[i]
            # accept/reject death
            S = rand()<1/r(u,Sprop) ? Sprop : S 
        end
    end
    return S
end


# function sample_pp(rng::Random.AbstractRNG,
#     p::GibbsProcess{<:Function}, 
#     b::Box; niter=10_000)

#     # region configuration
#     lo, up = coordinates.(extrema(b))
#     V = measure(b)

#     # Number of points
#     N(S) = length(S.items)

#     # product of uniform distributions
#     U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])
#     I(n) = DiscreteUniform(1,n) 

#     # probabilities
#     logλ(u,S) = N(S)>0 ? p.logλ(u,S) : 0
#     r(u,S) = exp(logλ(u,S)+log(V)-log(N(S)+1))
#     pbirth = 0.5
    
#     S = sample_pp(BinomialProcess(15),b)
#     #X = s₀.items 

#     for m in 1:niter
#         Sprop = deepcopy(S)
#         if rand() < pbirth
#             u = Point(rand(U))
#             Sprop = Sprop ∪ u
#             # accept/reject birth
#             S = rand()<r(u,S) ? Sprop : S 
#         elseif N(S)>1
#             # randomly select and remove point
#             i = rand(I(N(S)))
#             deleteat!(Sprop.items,i)
#             u = S.items[i]
#             # accept/reject death
#             S = rand()<1/r(u,Sprop) ? Sprop : S 
#         end
#     end
#     return S
# end

# function sample_pp(rng::Random.AbstractRNG,
#     p::GibbsProcess{<:Function}, 
#     b::Box; niter=10_000)

#     # region configuration
#     lo, up = coordinates.(extrema(b))
#     V = measure(b)

#     # Number of points
#     N(X) = length(X)

#     # product of uniform distributions
#     U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])
#     I(n) = DiscreteUniform(1,n) 

#     # probabilities
#     logλ(u,X) = N(X)>0 ? p.logλ(u,X) : 0
#     r(u,X) = exp(logλ(u,X)+log(V)-log(N(X)+1))
#     pbirth = 0.5
    
#     s₀ = sample_pp(BinomialProcess(15),b)
#     X = s₀.items 

#     for m in 1:niter
#         Xprop = deepcopy(X)
#         if rand() < pbirth
#             u = Point(rand(U))
#             Xprop = vcat(Xprop,u)
#             # accept/reject birth
#             X = rand()<r(u,X) ? Xprop : X 
#         elseif N(X)>1
#             # randomly select and remove point
#             i = rand(I(N(X)))
#             deleteat!(Xprop,i)
#             # accept/reject death
#             X = rand()<1/r(X[i],Xprop) ? Xprop : X 
#         end
#     end
#     s = PointSet(X)
#     return s
# end











#--------------------
# Shorthand sample
#--------------------

"""
   sample_pp(p,b)
Sample a point pattern from point process 'p' over box region 'b'.
"""
sample_pp(p::PointProcess, b::Box{Dim,T}) where {Dim,T} = sample_pp(Random.GLOBAL_RNG, p, b)



###### Sample on linear networks
###### Sample on linear networks
function sample_pp(p::BinomialProcess,L::LinearNetwork)
    probs = Meshes.measure.(L.segments)./measureL(L)
    pts = Array{Point2}(undef,p.n)
    segids = Array{Int64}(undef,p.n)
    for i in 1:p.n
        segids[i] = sample(1:length(L.segments), Weights(probs))
        pts[i] = SamplePointOnSegment(L.segments[segids[i]])
    end
    return PointSet(pts), segids
end

#= function sample_pp(p::BinomialProcess,L::LinearNetwork)
    probs = Meshes.measure.(L.segments)./measure(L)
    pts = Array{Point2}(undef,p.n)
    for i in 1:p.n
        s = sample(L.segments, Weights(probs))
        pts[i] = SamplePointOnSegment(s)
    end
    return PointSet(pts)
end
 =#
function sample_pp(p::GibbsProcess{<:Function}, 
    L::LinearNetwork; niter=10_000, progress=false)

    # region configuration
    #lo, up = coordinates.(extrema(b))
    #V = measure(b)

    # Network configuration
    V = measureL(L)
    probs = Meshes.measure.(L.segments)./V
    function sample_from_network() 
        segid = sample(1:length(L.segments), Weights(probs))
        #s = sample(L.segments, Weights(probs))
        point = SamplePointOnSegment(L.segments[segid])
        return point, segid
    end

    # Number of points
    N(S) = length(S.items)

    # product of uniform distributions
    #U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])
    I(n) = DiscreteUniform(1,n) 

    # probabilities
    logλ(u,S) = N(S)>0 ? p.logλ(u,S) : 0
    r(u,S) = exp(logλ(u,S)+log(V)-log(N(S)+1))
    pbirth = 0.5
    pmove = 0.5
    rmove(u,ξ,S₋) = pmove*exp(logλ(u,S₋)-logλ(ξ,S₋))/(N(S₋)+1)

    # Initial sample
    S, segids = sample_pp(BinomialProcess(100),L)
    #println("Performing ",niter," steps")
    for m in 1:niter
        if progress
            if mod(m,100)==0
                println("step ",m)
            end
        end
        Sprop = deepcopy(S)
        if rand() < pmove
            # Randomly select and move point
            i = rand(I(N(S)))
            ξ = Sprop.items[i]
            deleteat!(Sprop.items,i)
            u,segid = sample_from_network()
            # accept/reject move
            if rand()<rmove(u,ξ,Sprop)
                S = Sprop ∪ u
                deleteat!(segids,i)
                push!(segids,segid)
            end
            #S = rand()<rmove(u,ξ,Sprop) ? Sprop ∪ u : S  # EDIT THIS!
        elseif rand() < pbirth
            # Randomly generate new point
            u,segid = sample_from_network()
            Sprop = Sprop ∪ u
            # accept/reject birth
            if rand()<r(u,S)
                S = Sprop
                push!(segids,segid)
            end
            #S = rand()<r(u,S) ? Sprop : S 
        elseif N(S)>1
            # randomly select and remove point
            i = rand(I(N(S)))
            deleteat!(Sprop.items,i)
            u = S.items[i]
            # accept/reject death
            if rand()<1/r(u,Sprop)
                S = Sprop
                deleteat!(segids,i)
            end
            #S = rand()<1/r(u,Sprop) ? Sprop : S 
        end
    end
    return S, segids
end




#= function sample_pp(p::GibbsProcess{<:Function}, 
    L::LinearNetwork; niter=10_000, progress=false)

    # region configuration
    #lo, up = coordinates.(extrema(b))
    #V = measure(b)

    # Network configuration
    V = measure(L)
    probs = Meshes.measure.(L.segments)./V
    function sample_from_network() 
        s = sample(L.segments, Weights(probs))
        point = SamplePointOnSegment(s)
        return point
    end

    # Number of points
    N(S) = length(S.items)

    # product of uniform distributions
    #U = product_distribution([Uniform(lo[i], up[i]) for i in 1:embeddim(b)])
    I(n) = DiscreteUniform(1,n) 

    # probabilities
    logλ(u,S) = N(S)>0 ? p.logλ(u,S) : 0
    r(u,S) = exp(logλ(u,S)+log(V)-log(N(S)+1))
    pbirth = 0.5
    pmove = 0.5
    rmove(u,ξ,S₋) = pmove*exp(logλ(u,S₋)-logλ(ξ,S₋))/(N(S₋)+1)

    # Initial sample
    S = sample_pp(BinomialProcess(15),L)
    #println("Performing ",niter," steps")
    for m in 1:niter
        if progress
            if mod(m,100)==0
                println("step ",m)
            end
        end
        Sprop = deepcopy(S)
        if rand() < pmove
            # Randomly select and move point
            i = rand(I(N(S)))
            ξ = Sprop.items[i]
            deleteat!(Sprop.items,i)
            u = sample_from_network()
            # accept/reject move
            S = rand()<rmove(u,ξ,Sprop) ? Sprop ∪ u : S  # EDIT THIS!
        elseif rand() < pbirth
            # Randomly generate new point
            u = sample_from_network()
            Sprop = Sprop ∪ u
            # accept/reject birth
            S = rand()<r(u,S) ? Sprop : S 
        elseif N(S)>1
            # randomly select and remove point
            i = rand(I(N(S)))
            deleteat!(Sprop.items,i)
            u = S.items[i]
            # accept/reject death
            S = rand()<1/r(u,Sprop) ? Sprop : S 
        end
    end
    return S
end
 =#



#--------------------
# ToDo
#--------------------

### Function to sample from polygon, P
## - triangulate polygon -> [T1,...,T_k]
## - weight triangles by relative volume -> pi = measure(Ti)/measure(P)
## - Sample triangle Ti from the set of triangles based on weights pi
## - Sample uniformly from Ti (there are good algorithms)
