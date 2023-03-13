function TVsplit(s::PointSet, p::Real)
    @assert (p>0)&(p<1) "Prob. must satisfy 0<p<1"
    @assert length(s.items)>0 "PointSet must not be empty"
    idxT = [rand()<p for _ in eachindex(s.items)]
    sT = PointSet(s.items[idxT])
    sV = PointSet(setdiff(s.items,sT.items))
    return (sT, sV)
end

function Loss(TVset::Vector{Tuple{PointSet{2, Float64}, PointSet{2, Float64}}},
    logλ::Function,b::Box,p::Real;independent=false)
    k = length(TVset)
    I = zeros(k)
    for i in 1:k
        sT, sV = TVset[i]
        I[i] = PredictionError(sT,sV,logλ,b;p=p,independent=independent)
    end
    return mean(abs(I[i]) for i in 1:k)
end


function EstimateParams(m::model_pp, s::PointSet, b::Box; p=0.8, k=20)
    # Create CV folds
    CVfolds = [TVsplit(s,p) for i in 1:k]
    # Loss function
    L(θ) = Loss(CVfolds, (u,s)->m.logλ(u,s,θ),b,p; independent=m.independent)
    # Find minimum
    #res = optimize(L, m.lo, m.up, m.θ)
    res = optimize(L, m.lo, m.up, m.θ, Fminbox(NelderMead()),
            Optim.Options(x_tol = 1e-8,
            iterations = 1000,
            store_trace = false,
            show_trace = true,
            time_limit = 15))
    if Optim.converged(res)==true
        return Optim.minimizer(res)
    else
        return println("Optimization did not converge.")
    end
end

function EstimateParamsPL(m::model_pp, s::PointSet, b::Box)
    lo, up = coordinates.(extrema(b))
    println("Computing integral over cond. int.")
    function integral(θ::Vector)
        (val,err) = hcubature(u->exp(m.logλ(Point(u),s,θ)), lo, up;
                      reltol=1e-8, abstol=1e-12, maxevals=100_000)
        return val
    end
    loglik(θ::Vector) = sum(m.logλ(x,s\PointSet(x),θ) for x in s.items) - integral(θ)
    println("Optimizing log-likelihood")
    res = optimize(θ->-loglik(θ), m.lo, m.up, m.θ, Fminbox(NelderMead()),
            Optim.Options(x_tol = 1e-8,
            iterations = 1000,
            store_trace = false,
            show_trace = true,
            time_limit = 15))
    if Optim.converged(res)==true
        return Optim.minimizer(res)
    else
        return println("Optimization did not converge.")
    end
end

function EstimateParamsPL_Pois(S::Vector{Function}, s::PointSet, W::Box; nd = 80)
    boxes = partitions(W,nd)
    u = PointSet([centroid(box) for box in boxes])
    sw = PointSet(filter(x->x∈W,s.items))
    v = u ∪ sw
    Nv = length(v.items)
    K = length(S)
    boxind = [findfirst(box -> in(x,box),boxes) for x in v.items] 
    a = measure(W)/(nd*nd)
    # Now define
    w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        box = boxes[boxind[i]]
        vi = v.items[i]
        w[i] = a / (N(v,box))
        y[i] = (vi ∈ sw.items) ? 1/w[i] : 0
        for j in 1:K
            X[i,j] = S[j](vi,s\PointSet(vi))
            #X[i,j] = S[j](vi,s)
        end
    end
    res = glm(X,y,Poisson(),LogLink(); wts=w)
    return res
end


function EstimateParamsPL_Logit(S::Vector{Function}, s::PointSet, W::Box; nd = 80)
    boxes = partitions(W,nd)
    u = PointSet([centroid(box) for box in boxes])
    sw = PointSet(filter(x->x∈W,s.items))
    v = u ∪ sw
    Nv = length(v.items)
    K = length(S)
    boxind = [findfirst(box -> in(x,box),boxes) for x in v.items] 
    r = (nd*nd)/measure(W)
    ofs = fill(log(1/r),Nv)
    # Now define
    #w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        box = boxes[boxind[i]]
        vi = v.items[i]
        #w[i] = a / (N(v,box))
        y[i] = (vi ∈ sw.items) ? 1 : 0
        for j in 1:K
            X[i,j] = S[j](vi,s\PointSet(vi))
            #X[i,j] = S[j](vi,s)
        end
    end
    res = glm(X,y,Bernoulli(),LogitLink(); offset=ofs)
    return res
end

suggest_nd(s,W) = ceil(Int64,sqrt(4*N(s,W)))

function EstimateParamsPL_Logit(S::Vector{Function}, s::PointSet, L::LinearNetwork; nd = 400)
    segments = L.segments
    probs = Meshes.measure.(segments)./measure(L)
    nu = floor.(Int64,probs.*nd)
    u = PointSet(vcat([SamplePointsOnSegment(segments[i],nu[i]) for i in 1:length(segments)]...))
    #segids_u = vcat([repeat(i,nu[i]) for i in 1:length(segments)]...)
    #sw = PointSet(filter(x->x∈W,s.items))
    #v = u ∪ sw
    v = PointSet(vcat(u.items,s.items))
    #segids = vcat(segids_u,segids)
    Nv = length(v.items)
    K = length(S)
    #segind = [findfirst(seg -> in(x,seg),segments) for x in v.items] 
    r = nd/measure(L)
    ofs = fill(log(1/r),Nv)
    # Now define
    #w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        #seg = segments[segids[i]]
        vi = v.items[i]
        #w[i] = a / (N(v,box))
        y[i] = (vi ∈ s.items) ? 1 : 0
        for j in 1:K
            X[i,j] = S[j](vi,s\PointSet(vi))
            #X[i,j] = S[j](vi,s)
        end
    end
    res = glm(X,y,Bernoulli(),LogitLink(); offset=ofs)
    return res
end

function EstimateParamsPL_Logit2(S::Vector{Function}, s::PointSet, L::LinearNetwork, W::Box; nd = 80)
    boxes = partitions(W,nd)
    networks = [subnetwork(L,b) for b in boxes]
    u = PointSet([SamplePointOnNetwork(net) for net in networks])
    sw = PointSet(filter(x->x∈W,s.items))
    v = u ∪ sw
    Nv = length(v.items)
    K = length(S)
    boxind = [findfirst(box -> in(x,box),boxes) for x in v.items] 
    r = (nd*nd)/measure(L)
    ofs = fill(log(1/r),Nv)
    # Now define
    #w = zeros(Nv)
    y = zeros(Nv)
    X = zeros(Nv,K)
    for i in 1:Nv
        box = boxes[boxind[i]]
        vi = v.items[i]
        #w[i] = a / (N(v,box))
        y[i] = (vi ∈ sw.items) ? 1 : 0
        for j in 1:K
            X[i,j] = S[j](vi,s\PointSet(vi))
            #X[i,j] = S[j](vi,s)
        end
    end
    res = glm(X,y,Bernoulli(),LogitLink(); offset=ofs)
    return res
end